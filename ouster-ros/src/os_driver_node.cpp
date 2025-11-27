/**
 * Copyright (c) 2018-2023, Ouster, Inc.
 * All rights reserved.
 *
 * @file os_driver.cpp
 * @brief This node combines the capabilities of os_sensor, os_cloud and
 * os_image into a single ROS node/component
 */

// prevent clang-format from altering the location of "ouster_ros/os_ros.h", the
// header file needs to be the first include due to PCL_NO_PRECOMPILE flag
// clang-format off
#include "ouster_ros/os_ros.h"
// clang-format on

#include "os_sensor_node.h"

#include "os_static_transforms_broadcaster.h"
#include "imu_packet_handler.h"
#include "lidar_packet_handler.h"
#include "point_cloud_processor.h"
#include "laser_scan_processor.h"
#include "image_processor.h"
#include "point_cloud_processor_factory.h"
#include "telemetry_handler.h"

#include <point_cloud_transport/point_cloud_transport.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2/time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/io.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <vector>
#include <stdexcept>

namespace ouster_ros {

namespace sensor = ouster::sensor;
using ouster::sensor::LidarPacket;
using ouster::sensor::ImuPacket;
using ouster::sensor::LidarPacket;

namespace {

bool is_ros_time_mode(const std::string& mode) {
    return mode == "TIME_FROM_ROS_TIME" || mode == "TIME_FROM_ROS_RECEPTION";
}

double compute_scan_col_ts_spacing_ns(sensor::lidar_mode ld_mode) {
    const auto scan_width = sensor::n_cols_of_lidar_mode(ld_mode);
    const auto scan_frequency = sensor::frequency_of_lidar_mode(ld_mode);
    constexpr double kOneSecondInNs = 1e9;
    return scan_width > 0 && scan_frequency > 0
               ? kOneSecondInNs / (scan_width * scan_frequency)
               : 0.0;
}

uint16_t packet_col_index(const sensor::packet_format& pf,
                          const uint8_t* lidar_buf) {
    return pf.col_measurement_id(pf.nth_col(0, lidar_buf));
}

uint64_t first_nonzero_timestamp(
    const ouster::LidarScan::Header<uint64_t>& ts_header) {
    auto it =
        std::find_if(ts_header.data(), ts_header.data() + ts_header.size(),
                     [](uint64_t ts) { return ts != 0; });
    return it == ts_header.data() + ts_header.size() ? 0 : *it;
}

}  // namespace

struct SensorProfileConfig {
    std::string name;
    std::string sensor_hostname;
    std::string udp_dest;
    std::string mtp_dest;
    bool mtp_main{false};
    int lidar_port{0};
    int imu_port{0};
    std::string lidar_mode;
    std::string timestamp_mode;
    std::string udp_profile_lidar;
    double ptp_utc_tai_offset{-37.0};
    int azimuth_window_start{0};
    int azimuth_window_end{360000};
    std::string metadata_path;
    std::string point_cloud_frame;
};

struct SensorContext {
    SensorProfileConfig profile;
    sensor::sensor_config config;
    std::shared_ptr<sensor::client> client;
    sensor::sensor_info info;
    LidarScanProcessor lidar_scan_processor;
    std::unique_ptr<ouster::ScanBatcher> scan_batcher;
    std::unique_ptr<ouster::LidarScan> active_scan;
    const sensor::packet_format* packet_format{nullptr};
    sensor::LidarPacket lidar_packet;
    std::unique_ptr<std::thread> packet_thread;
    std::atomic<bool> running{false};
    std::mutex scan_mutex;
    uint64_t active_scan_ts{0};
    rclcpp::Time active_scan_stamp;
    bool stamp_ready{false};
    bool expect_frame_start{true};
    double scan_col_ts_spacing_ns{0.0};
    int64_t ptp_utc_tai_offset_ns{0};
    double min_scan_valid_columns_ratio{0.0};
    std::atomic<bool> scan_dirty{false};
    std::mutex clouds_mutex;
    std::vector<sensor_msgs::msg::PointCloud2> processed_clouds;
    rclcpp::Time last_cloud_stamp;
};

class OusterDriver : public OusterSensor {
   public:
    OUSTER_ROS_PUBLIC
    explicit OusterDriver(const rclcpp::NodeOptions& options)
        : OusterSensor("os_driver", options), tf_bcast(this) {
        tf_bcast.declare_parameters();
        tf_bcast.parse_parameters();
        declare_parameter("proc_mask", "IMU|PCL|SCAN|IMG|RAW|TLM");
        declare_parameter("scan_ring", 0);
        declare_parameter("ptp_utc_tai_offset", -37.0);
        declare_parameter("point_type", "original");
        declare_parameter("organized", true);
        declare_parameter("destagger", true);
        declare_parameter("min_range", 0.0);
        declare_parameter("max_range", 1000.0);
        declare_parameter("v_reduction", 1);
        declare_parameter("min_scan_valid_columns_ratio", 0.0);
        declare_parameter("mask_path", "");
        declare_parameter("sensor_profiles",
                          std::vector<std::string>{});
        declare_parameter("fusion_target_frame", "os_fusion");
        declare_parameter("fusion_publish_rate", 20.0);
        declare_parameter("fusion_tf_timeout", 0.1);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
            *tf_buffer_);
    }

    ~OusterDriver() override {
        RCLCPP_DEBUG(get_logger(), "OusterDriver::~OusterDriver() called");
    }

    virtual LifecycleNodeInterface::CallbackReturn on_configure(
        const rclcpp_lifecycle::State& state) override {
        LifecycleNode::on_configure(state);
        sensor_profile_names_ =
            get_parameter("sensor_profiles").as_string_array();
        fusion_mode_enabled_ = !sensor_profile_names_.empty();
        if (!fusion_mode_enabled_) {
            return OusterSensor::on_configure(state);
        }

        for (const auto& profile : sensor_profile_names_) {
            declare_profile_parameters(profile);
        }

        if (!initialize_fusion_mode()) {
            RCLCPP_ERROR(get_logger(),
                         "Failed to initialize fusion driver configuration");
            return LifecycleNodeInterface::CallbackReturn::FAILURE;
        }

        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    virtual LifecycleNodeInterface::CallbackReturn on_activate(
        const rclcpp_lifecycle::State& state) override {
        if (!fusion_mode_enabled_) {
            return OusterSensor::on_activate(state);
        }
        LifecycleNode::on_activate(state);

        if (!fusion_timer_) {
            double rate = fusion_publish_rate_ > 0.0 ? fusion_publish_rate_ : 1.0;
            auto period = std::chrono::duration<double>(1.0 / rate);
            fusion_timer_ = create_wall_timer(
                period, std::bind(&OusterDriver::fusion_timer_callback, this));
            RCLCPP_INFO(get_logger(),
                        "Fusion timer started at %.2f Hz targeting frame '%s'",
                        rate, fusion_frame_.c_str());
        }

        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    virtual LifecycleNodeInterface::CallbackReturn on_deactivate(
        const rclcpp_lifecycle::State& state) override {
        if (!fusion_mode_enabled_) {
            return OusterSensor::on_deactivate(state);
        }
        LifecycleNode::on_deactivate(state);
        if (fusion_timer_) {
            fusion_timer_->cancel();
            fusion_timer_.reset();
        }
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    virtual LifecycleNodeInterface::CallbackReturn on_cleanup(
        const rclcpp_lifecycle::State& state) override {
        if (!fusion_mode_enabled_) {
            return OusterSensor::on_cleanup(state);
        }
        LifecycleNode::on_cleanup(state);
        if (fusion_timer_) {
            fusion_timer_->cancel();
            fusion_timer_.reset();
        }
        stop_sensor_contexts();
        fusion_pubs_.clear();
        sensor_contexts_.clear();
        pct_node_.reset();
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    virtual void on_metadata_updated(const sensor::sensor_info& info) override {
        OusterSensor::on_metadata_updated(info);
        if (tf_bcast.publish_static_tf()) {
          tf_bcast.broadcast_transforms(info);
        }
    }

    virtual void create_publishers() override {
        auto proc_mask = get_parameter("proc_mask").as_string();
        auto tokens = impl::parse_tokens(proc_mask, '|');

        bool use_system_default_qos =
            get_parameter("use_system_default_qos").as_bool();
        rclcpp::QoS system_default_qos = rclcpp::SystemDefaultsQoS();
        rclcpp::QoS sensor_data_qos = rclcpp::SensorDataQoS();
        auto selected_qos =
            use_system_default_qos ? system_default_qos : sensor_data_qos;

        auto timestamp_mode = get_parameter("timestamp_mode").as_string();
        auto ptp_utc_tai_offset =
            get_parameter("ptp_utc_tai_offset").as_double();

        if (impl::check_token(tokens, "IMU")) {
            imu_pub =
                create_publisher<sensor_msgs::msg::Imu>("imu", selected_qos);
            imu_packet_handler = ImuPacketHandler::create(
                info, tf_bcast.imu_frame_id(), timestamp_mode,
                static_cast<int64_t>(ptp_utc_tai_offset * 1e+9));
        }

        auto min_scan_valid_columns_ratio = get_parameter("min_scan_valid_columns_ratio").as_double();
        if (min_scan_valid_columns_ratio < 0.0f || min_scan_valid_columns_ratio > 1.0f) {
            RCLCPP_FATAL(get_logger(), "min_scan_valid_columns_ratio needs to be in the range [0, 1]");
            throw std::runtime_error("min_scan_valid_columns_ratio out of bounds!");
        }

        auto mask_path = get_parameter("mask_path").as_string();

        int num_returns = get_n_returns(info);

        std::vector<LidarScanProcessor> processors;
        if (impl::check_token(tokens, "PCL")) {
            lidar_pubs.resize(num_returns);

            // extra node until point_cloud_transport supports LifecycleNode
            // https://github.com/ros-perception/point_cloud_transport/pull/109
            auto pct_node = std::make_shared<rclcpp::Node>("os_driver_point_cloud_transport");
            point_cloud_transport::PointCloudTransport pct(pct_node);

            for (int i = 0; i < num_returns; ++i) {
                std::string topic = topic_for_return("points", i);
                // Convert rclcpp::QoS to rmw_qos_profile_t
                lidar_pubs[i] = std::make_shared<point_cloud_transport::Publisher>(
                    pct.advertise(topic, selected_qos.get_rmw_qos_profile()));
            }

            auto point_type = get_parameter("point_type").as_string();
            auto organized = get_parameter("organized").as_bool();
            auto destagger = get_parameter("destagger").as_bool();
            auto min_range_m = get_parameter("min_range").as_double();
            auto max_range_m = get_parameter("max_range").as_double();
            if (min_range_m < 0.0 || max_range_m < 0.0) {
                RCLCPP_FATAL(get_logger(), "min_range and max_range need to be positive");
                throw std::runtime_error("negative range limits!");
            }
            if (min_range_m >= max_range_m) {
                const auto error_msg = "min_range can't be equal or exceed max_range";
                RCLCPP_FATAL(get_logger(), error_msg);
                throw std::runtime_error(error_msg);
            }
            // convert to millimeters
            uint32_t min_range = impl::ulround(min_range_m * 1000);
            uint32_t max_range = impl::ulround(max_range_m * 1000);
            auto v_reduction = get_parameter("v_reduction").as_int();
            auto valid_values = std::vector<int>{1, 2, 4, 8, 16};
            if (std::find(valid_values.begin(), valid_values.end(),
                          v_reduction) == valid_values.end()) {
                RCLCPP_FATAL(get_logger(),
                    "v_reduction needs to be one of the values: {1, 2, 4, 8, 16}");
                throw std::runtime_error("invalid v_reduction value!");
            }

            processors.push_back(
                PointCloudProcessorFactory::create_point_cloud_processor(point_type,
                    info, tf_bcast.point_cloud_frame_id(),
                    tf_bcast.apply_lidar_to_sensor_transform(),
                    organized, destagger, min_range, max_range, v_reduction, mask_path,
                    [this](PointCloudProcessor_OutputType msgs) {
                        for (size_t i = 0; i < msgs.size(); ++i)
                            lidar_pubs[i]->publish(*msgs[i]);
                    }
                )
            );

            // warn about profile incompatibility
            if (PointCloudProcessorFactory::point_type_requires_intensity(point_type) &&
                !PointCloudProcessorFactory::profile_has_intensity(info.format.udp_profile_lidar)) {
                RCLCPP_WARN_STREAM(
                    get_logger(),
                    "selected point type '" << point_type
                    << "' is not compatible with the udp profile: "
                    << to_string(info.format.udp_profile_lidar));
            }
        }

        if (impl::check_token(tokens, "SCAN")) {
            scan_pubs.resize(num_returns);
            for (int i = 0; i < num_returns; ++i) {
                scan_pubs[i] = create_publisher<sensor_msgs::msg::LaserScan>(
                    topic_for_return("scan", i), selected_qos);
            }

            // TODO: avoid duplication in os_cloud_node
            int beams_count = static_cast<int>(get_beams_count(info));
            int scan_ring = get_parameter("scan_ring").as_int();
            scan_ring = std::min(std::max(scan_ring, 0), beams_count - 1);
            if (scan_ring != get_parameter("scan_ring").as_int()) {
                RCLCPP_WARN_STREAM(
                    get_logger(),
                    "scan ring is set to a value that exceeds available range"
                    "please choose a value between [0, "
                        << beams_count
                        << "], "
                           "ring value clamped to: "
                        << scan_ring);
            }

            processors.push_back(LaserScanProcessor::create(
                info, tf_bcast.lidar_frame_id(), scan_ring,
                [this](LaserScanProcessor::OutputType msgs) {
                    for (size_t i = 0; i < msgs.size(); ++i) scan_pubs[i]->publish(*msgs[i]);
                }));
        }

        if (impl::check_token(tokens, "IMG")) {
            const std::map<sensor::ChanField, std::string>
                channel_field_topic_map_1{
                    {sensor::ChanField::RANGE, "range_image"},
                    {sensor::ChanField::SIGNAL, "signal_image"},
                    {sensor::ChanField::REFLECTIVITY, "reflec_image"},
                    {sensor::ChanField::NEAR_IR, "nearir_image"}};

            const std::map<sensor::ChanField, std::string>
                channel_field_topic_map_2{
                    {sensor::ChanField::RANGE, "range_image"},
                    {sensor::ChanField::SIGNAL, "signal_image"},
                    {sensor::ChanField::REFLECTIVITY, "reflec_image"},
                    {sensor::ChanField::NEAR_IR, "nearir_image"},
                    {sensor::ChanField::RANGE2, "range_image2"},
                    {sensor::ChanField::SIGNAL2, "signal_image2"},
                    {sensor::ChanField::REFLECTIVITY2, "reflec_image2"}};

            auto which_map = num_returns == 1 ? &channel_field_topic_map_1
                                              : &channel_field_topic_map_2;
            for (auto it = which_map->begin(); it != which_map->end(); ++it) {
                image_pubs[it->first] =
                    create_publisher<sensor_msgs::msg::Image>(it->second,
                                                              selected_qos);
            }

            processors.push_back(ImageProcessor::create(
                info, tf_bcast.point_cloud_frame_id(), mask_path,
                [this](ImageProcessor::OutputType msgs) {
                    for (auto it = msgs.begin(); it != msgs.end(); ++it) {
                        image_pubs[it->first]->publish(*it->second);
                    }
                }));
        }

        if (impl::check_token(tokens, "PCL") || impl::check_token(tokens, "SCAN") ||
            impl::check_token(tokens, "IMG"))
            lidar_packet_handler = LidarPacketHandler::create(
                info, processors, timestamp_mode,
                static_cast<int64_t>(ptp_utc_tai_offset * 1e+9),
                min_scan_valid_columns_ratio);

        if (impl::check_token(tokens, "TLM")) {
            telemetry_pub =
                create_publisher<ouster_sensor_msgs::msg::Telemetry>("telemetry",
                                                                     selected_qos);
            telemetry_handler = TelemetryHandler::create(
                info, timestamp_mode,
                static_cast<int64_t>(ptp_utc_tai_offset * 1e+9));
        }

        publish_raw = impl::check_token(tokens, "RAW");
        if (publish_raw)
            OusterSensor::create_publishers();
    }

    virtual void on_lidar_packet_msg(const LidarPacket& lidar_packet) override {
        if (telemetry_handler) {
            auto telemetry = telemetry_handler(lidar_packet);
            telemetry_pub->publish(telemetry);
        }

        if (lidar_packet_handler)
            lidar_packet_handler(lidar_packet);

        if (publish_raw)
            OusterSensor::on_lidar_packet_msg(lidar_packet);
    }

    virtual void on_imu_packet_msg(const ImuPacket& imu_packet) override {
        if (imu_packet_handler)
            imu_pub->publish(imu_packet_handler(imu_packet));

        if (publish_raw)
            OusterSensor::on_imu_packet_msg(imu_packet);
    }

    virtual void cleanup() override {
        imu_packet_handler = nullptr;
        lidar_packet_handler = nullptr;
        imu_pub.reset();
        for (auto p : lidar_pubs) p.reset();
        for (auto p : scan_pubs) p.reset();
        for (auto p : image_pubs) p.second.reset();
        OusterSensor::cleanup();
    }

   private:
    OusterStaticTransformsBroadcaster<rclcpp_lifecycle::LifecycleNode> tf_bcast;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
    std::vector<std::shared_ptr<point_cloud_transport::Publisher>>
        lidar_pubs;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr>
        scan_pubs;
    std::map<sensor::ChanField,
             rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
        image_pubs;
    ImuPacketHandler::HandlerType imu_packet_handler;
    LidarPacketHandler::HandlerType lidar_packet_handler;

    bool publish_raw = false;

    rclcpp::Publisher<ouster_sensor_msgs::msg::Telemetry>::SharedPtr telemetry_pub;
TelemetryHandler::HandlerType telemetry_handler;

// Multi-sensor fusion members
bool fusion_mode_enabled_{false};
std::vector<std::shared_ptr<SensorContext>> sensor_contexts_;
    std::vector<std::shared_ptr<point_cloud_transport::Publisher>> fusion_pubs_;
    rclcpp::TimerBase::SharedPtr fusion_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<rclcpp::Node> pct_node_;
    std::string fusion_frame_;
    double fusion_publish_rate_{20.0};
    double fusion_tf_timeout_{0.1};
    std::vector<std::string> sensor_profile_names_;

    void declare_profile_parameters(const std::string& profile);
    SensorProfileConfig load_profile_config(const std::string& profile);
    sensor::sensor_config build_sensor_config(
        const SensorProfileConfig& profile);
    bool initialize_fusion_mode();
    bool start_sensor_context(const std::shared_ptr<SensorContext>& ctx);
    void stop_sensor_contexts();
    void fusion_timer_callback();
    bool ensure_fusion_publishers(size_t num_returns,
                                  rclcpp::QoS qos_profile);
    void handle_streaming_packet(const std::shared_ptr<SensorContext>& ctx,
                                 const sensor::LidarPacket& packet);
    void update_scan_timestamps_locked(
        const std::shared_ptr<SensorContext>& ctx,
        const rclcpp::Time& packet_receive_time, int packet_first_col);
    bool snapshot_active_scan(const std::shared_ptr<SensorContext>& ctx,
                              ouster::LidarScan& scan_copy,
                              uint64_t& scan_ts, rclcpp::Time& msg_ts);
    rcl_clock_type_t node_clock_type() const {
        return get_clock()->get_clock_type();
    }
    bool collect_transformed_clouds(
        const std::shared_ptr<SensorContext>& ctx,
        std::vector<sensor_msgs::msg::PointCloud2>& transformed,
        rclcpp::Time& msg_ts);
    bool transform_cloud(const sensor_msgs::msg::PointCloud2& in,
                         sensor_msgs::msg::PointCloud2& out,
                         const std::string& frame, const rclcpp::Time& stamp);
    sensor_msgs::msg::PointCloud2 fuse_transformed_clouds(
        const std::vector<sensor_msgs::msg::PointCloud2>& clouds,
        const std::string& frame, const rclcpp::Time& stamp);
};

void OusterDriver::declare_profile_parameters(const std::string& profile) {
    auto prefix = profile + ".";
    auto declare_string = [this](const std::string& name,
                                 const std::string& default_value) {
        if (!has_parameter(name)) declare_parameter(name, default_value);
    };
    auto declare_int = [this](const std::string& name, int default_value) {
        if (!has_parameter(name)) declare_parameter(name, default_value);
    };
    auto declare_bool = [this](const std::string& name, bool default_value) {
        if (!has_parameter(name)) declare_parameter(name, default_value);
    };
    auto declare_double = [this](const std::string& name, double default_value) {
        if (!has_parameter(name)) declare_parameter(name, default_value);
    };

    declare_string(prefix + "sensor_hostname", "");
    declare_string(prefix + "udp_dest", "");
    declare_string(prefix + "mtp_dest", "");
    declare_bool(prefix + "mtp_main", false);
    declare_int(prefix + "lidar_port", 0);
    declare_int(prefix + "imu_port", 0);
    declare_string(prefix + "lidar_mode", "");
    declare_string(prefix + "timestamp_mode", "");
    declare_string(prefix + "udp_profile_lidar", "");
    declare_int(prefix + "azimuth_window_start", 0);
    declare_int(prefix + "azimuth_window_end", 360000);
    declare_string(prefix + "metadata", "");
    declare_string(prefix + "point_cloud_frame", profile + "_os_lidar");
    declare_double(prefix + "ptp_utc_tai_offset", -37.0);
}

SensorProfileConfig OusterDriver::load_profile_config(
    const std::string& profile) {
    SensorProfileConfig cfg;
    cfg.name = profile;
    auto prefix = profile + ".";
    cfg.sensor_hostname = get_parameter(prefix + "sensor_hostname").as_string();
    cfg.udp_dest = get_parameter(prefix + "udp_dest").as_string();
    cfg.mtp_dest = get_parameter(prefix + "mtp_dest").as_string();
    cfg.mtp_main = get_parameter(prefix + "mtp_main").as_bool();
    cfg.lidar_port = get_parameter(prefix + "lidar_port").as_int();
    cfg.imu_port = get_parameter(prefix + "imu_port").as_int();
    cfg.lidar_mode = get_parameter(prefix + "lidar_mode").as_string();
    cfg.timestamp_mode = get_parameter(prefix + "timestamp_mode").as_string();
    if (!is_arg_set(cfg.timestamp_mode)) {
        cfg.timestamp_mode = get_parameter("timestamp_mode").as_string();
    }
    cfg.udp_profile_lidar =
        get_parameter(prefix + "udp_profile_lidar").as_string();
    cfg.azimuth_window_start =
        get_parameter(prefix + "azimuth_window_start").as_int();
    cfg.azimuth_window_end =
        get_parameter(prefix + "azimuth_window_end").as_int();
    cfg.metadata_path = get_parameter(prefix + "metadata").as_string();
    cfg.point_cloud_frame =
        get_parameter(prefix + "point_cloud_frame").as_string();
    cfg.ptp_utc_tai_offset =
        get_parameter(prefix + "ptp_utc_tai_offset").as_double();
    if (!is_arg_set(cfg.point_cloud_frame)) {
        cfg.point_cloud_frame = profile + "_os_lidar";
    }
    return cfg;
}

sensor::sensor_config OusterDriver::build_sensor_config(
    const SensorProfileConfig& profile) {
    sensor::sensor_config config;
    if (is_arg_set(profile.udp_dest)) config.udp_dest = profile.udp_dest;
    if (profile.lidar_port < 0 || profile.lidar_port > 65535)
        throw std::runtime_error("Invalid lidar port for profile " +
                                 profile.name);
    if (profile.imu_port < 0 || profile.imu_port > 65535)
        throw std::runtime_error("Invalid imu port for profile " +
                                 profile.name);
    if (profile.lidar_port) config.udp_port_lidar = profile.lidar_port;
    if (profile.imu_port) config.udp_port_imu = profile.imu_port;

    if (is_arg_set(profile.lidar_mode)) {
        auto mode = sensor::lidar_mode_of_string(profile.lidar_mode);
        if (!mode)
            throw std::runtime_error("Invalid lidar mode for profile " +
                                     profile.name);
        config.ld_mode = mode;
    }

    if (is_arg_set(profile.timestamp_mode) &&
        profile.timestamp_mode != "TIME_FROM_ROS_TIME" &&
        profile.timestamp_mode != "TIME_FROM_ROS_RECEPTION") {
        auto ts_mode =
            sensor::timestamp_mode_of_string(profile.timestamp_mode);
        if (!ts_mode)
            throw std::runtime_error("Invalid timestamp mode for profile " +
                                     profile.name);
        config.ts_mode = ts_mode;
    }

    if (is_arg_set(profile.udp_profile_lidar)) {
        auto udp_profile =
            sensor::udp_profile_lidar_of_string(profile.udp_profile_lidar);
        if (!udp_profile)
            throw std::runtime_error("Invalid udp profile for profile " +
                                     profile.name);
        config.udp_profile_lidar = udp_profile;
    }

    sensor::AzimuthWindow azm{profile.azimuth_window_start,
                              profile.azimuth_window_end};
    config.azimuth_window = azm;

    return config;
}

bool OusterDriver::start_sensor_context(
    const std::shared_ptr<SensorContext>& ctx) {
    try {
        if (!is_arg_set(ctx->profile.sensor_hostname)) {
            RCLCPP_ERROR(get_logger(), "Profile '%s' missing sensor hostname",
                         ctx->profile.name.c_str());
            return false;
        }

        uint8_t flags = sensor::CONFIG_FORCE_REINIT;
        sensor::set_config(ctx->profile.sensor_hostname, ctx->config, flags);

        std::shared_ptr<sensor::client> client;
        auto udp_dest = ctx->profile.udp_dest;
        if (sensor::in_multicast(udp_dest)) {
            client = sensor::mtp_init_client(ctx->profile.sensor_hostname,
                                             ctx->config, ctx->profile.mtp_dest,
                                             ctx->profile.mtp_main);
        } else if (ctx->profile.lidar_port != 0 &&
                   ctx->profile.imu_port != 0) {
            client = sensor::init_client(ctx->profile.sensor_hostname,
                                         ctx->profile.lidar_port,
                                         ctx->profile.imu_port);
        } else {
            client = sensor::init_client(
                ctx->profile.sensor_hostname, udp_dest, sensor::MODE_UNSPEC,
                sensor::TIME_FROM_UNSPEC, ctx->profile.lidar_port,
                ctx->profile.imu_port);
        }

        if (!client) {
            RCLCPP_ERROR(get_logger(),
                         "Failed to init client for profile '%s'",
                         ctx->profile.name.c_str());
            return false;
        }

        ctx->client = client;
        auto metadata = sensor::get_metadata(*client, 60, false);
        ctx->info = sensor::parse_metadata(metadata);
        if (is_arg_set(ctx->profile.metadata_path)) {
            write_text_to_file(ctx->profile.metadata_path, metadata);
        }
        RCLCPP_INFO(get_logger(),
                    "Profile '%s' connected (sn: %s, mode: %s)",
                    ctx->profile.name.c_str(), ctx->info.sn.c_str(),
                    sensor::to_string(ctx->info.mode).c_str());

        auto& pf = sensor::get_format(ctx->info);
        ctx->lidar_packet.buf.resize(pf.lidar_packet_size);
        ctx->packet_format = &pf;

        auto point_type = get_parameter("point_type").as_string();
        auto organized = get_parameter("organized").as_bool();
        auto destagger = get_parameter("destagger").as_bool();
        auto min_range_m = get_parameter("min_range").as_double();
        auto max_range_m = get_parameter("max_range").as_double();
        if (min_range_m < 0.0 || max_range_m < 0.0)
            throw std::runtime_error("Invalid range parameters");
        if (min_range_m >= max_range_m)
            throw std::runtime_error("min_range >= max_range");
        uint32_t min_range = impl::ulround(min_range_m * 1000);
        uint32_t max_range = impl::ulround(max_range_m * 1000);
        auto v_reduction = get_parameter("v_reduction").as_int();
        auto mask_path = get_parameter("mask_path").as_string();
        auto min_scan_valid_columns_ratio =
            get_parameter("min_scan_valid_columns_ratio").as_double();
        if (min_scan_valid_columns_ratio < 0.0 ||
            min_scan_valid_columns_ratio > 1.0)
            throw std::runtime_error(
                "min_scan_valid_columns_ratio out of bounds");
        ctx->min_scan_valid_columns_ratio = min_scan_valid_columns_ratio;

        auto ctx_ptr = ctx.get();
        ctx->lidar_scan_processor =
            PointCloudProcessorFactory::create_point_cloud_processor(
                point_type, ctx->info, ctx->profile.point_cloud_frame, false,
                organized, destagger, min_range, max_range, v_reduction,
                mask_path, [ctx_ptr](PointCloudProcessor_OutputType msgs) {
                    std::lock_guard<std::mutex> lock(ctx_ptr->clouds_mutex);
                    ctx_ptr->processed_clouds.resize(msgs.size());
                    for (size_t i = 0; i < msgs.size(); ++i) {
                        ctx_ptr->processed_clouds[i] = *msgs[i];
                    }
                    if (!msgs.empty()) {
                        ctx_ptr->last_cloud_stamp = msgs[0]->header.stamp;
                    }
                });

        if (PointCloudProcessorFactory::point_type_requires_intensity(point_type) &&
            !PointCloudProcessorFactory::profile_has_intensity(
                ctx->info.format.udp_profile_lidar)) {
            RCLCPP_WARN_STREAM(
                get_logger(),
                "Profile '" << ctx->profile.name
                              << "' selected point type '" << point_type
                              << "' which is incompatible with the udp profile "
                              << to_string(ctx->info.format.udp_profile_lidar));
        }

        ctx->scan_batcher = std::make_unique<ouster::ScanBatcher>(ctx->info);
        ctx->active_scan = std::make_unique<ouster::LidarScan>(
            ctx->info.format.columns_per_frame, ctx->info.format.pixels_per_column,
            ctx->info.format.udp_profile_lidar);
        ctx->scan_col_ts_spacing_ns =
            compute_scan_col_ts_spacing_ns(ctx->info.mode);
        ctx->ptp_utc_tai_offset_ns =
            static_cast<int64_t>(ctx->profile.ptp_utc_tai_offset * 1e9);
        ctx->expect_frame_start = true;
        ctx->stamp_ready = false;
        ctx->scan_dirty.store(false);

        ctx->running = true;
        ctx->packet_thread = std::make_unique<std::thread>([this, ctx]() {
            while (rclcpp::ok() && ctx->running.load()) {
                auto state = sensor::poll_client(*ctx->client);
                if (state == sensor::EXIT) break;
                if (state == sensor::TIMEOUT) continue;
                if (state & sensor::CLIENT_ERROR) {
                    RCLCPP_WARN(get_logger(),
                                "poll_client error for profile '%s'",
                                ctx->profile.name.c_str());
                    continue;
                }
                if (state & sensor::LIDAR_DATA) {
                    if (sensor::read_lidar_packet(*ctx->client,
                                                  ctx->lidar_packet)) {
                        handle_streaming_packet(ctx, ctx->lidar_packet);
                    }
                }
            }
            RCLCPP_INFO(get_logger(), "Profile '%s' packet thread exited",
                        ctx->profile.name.c_str());
        });
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(),
                     "Failed to start profile '%s': %s",
                     ctx->profile.name.c_str(), ex.what());
        return false;
    }

    return true;
}

bool OusterDriver::ensure_fusion_publishers(size_t num_returns,
                                            rclcpp::QoS qos_profile) {
    if (fusion_pubs_.size() == num_returns && pct_node_) return true;
    fusion_pubs_.clear();
    pct_node_ = std::make_shared<rclcpp::Node>("os_driver_fusion_pct");
    point_cloud_transport::PointCloudTransport pct(pct_node_);
    fusion_pubs_.reserve(num_returns);
    for (size_t i = 0; i < num_returns; ++i) {
        auto topic = topic_for_return("points", static_cast<int>(i));
        fusion_pubs_.push_back(std::make_shared<point_cloud_transport::Publisher>(
            pct.advertise(topic, qos_profile.get_rmw_qos_profile())));
    }
    return true;
}

bool OusterDriver::initialize_fusion_mode() {
    stop_sensor_contexts();
    sensor_contexts_.clear();

    fusion_frame_ = get_parameter("fusion_target_frame").as_string();
    fusion_publish_rate_ = get_parameter("fusion_publish_rate").as_double();
    fusion_tf_timeout_ = get_parameter("fusion_tf_timeout").as_double();

    for (const auto& profile_name : sensor_profile_names_) {
        auto cfg = load_profile_config(profile_name);
        auto ctx = std::make_shared<SensorContext>();
        ctx->profile = cfg;
        ctx->config = build_sensor_config(cfg);
        if (!start_sensor_context(ctx)) {
            stop_sensor_contexts();
            sensor_contexts_.clear();
            return false;
        }
        RCLCPP_INFO(get_logger(), "Profile '%s' streaming thread started",
                    profile_name.c_str());
        sensor_contexts_.push_back(ctx);
    }

    if (sensor_contexts_.empty()) {
        RCLCPP_ERROR(get_logger(), "No sensors configured for fusion mode");
        return false;
    }

    size_t num_returns = 0;
    for (const auto& ctx : sensor_contexts_) {
        num_returns = std::max(
            num_returns, static_cast<size_t>(get_n_returns(ctx->info)));
    }

    bool use_system_default_qos =
        get_parameter("use_system_default_qos").as_bool();
    rclcpp::QoS qos =
        use_system_default_qos ? rclcpp::QoS(rclcpp::SystemDefaultsQoS())
                               : rclcpp::QoS(rclcpp::SensorDataQoS());
    if (!ensure_fusion_publishers(num_returns, qos)) return false;

    return true;
}

void OusterDriver::stop_sensor_contexts() {
    for (auto& ctx : sensor_contexts_) {
        if (ctx->running) ctx->running = false;
        if (ctx->packet_thread && ctx->packet_thread->joinable()) {
            ctx->packet_thread->join();
        }
        ctx->packet_thread.reset();
        ctx->scan_batcher.reset();
        ctx->active_scan.reset();
        ctx->client.reset();
        ctx->packet_format = nullptr;
        ctx->lidar_scan_processor = nullptr;
        ctx->scan_dirty = false;
        ctx->stamp_ready = false;
        ctx->expect_frame_start = true;
        {
            std::lock_guard<std::mutex> lock(ctx->clouds_mutex);
            ctx->processed_clouds.clear();
        }
    }
}

void OusterDriver::update_scan_timestamps_locked(
    const std::shared_ptr<SensorContext>& ctx,
    const rclcpp::Time& packet_receive_time, int packet_first_col) {
    if (!ctx->active_scan) return;

    if (ctx->expect_frame_start && is_ros_time_mode(ctx->profile.timestamp_mode)) {
        const auto offset_ns = static_cast<int64_t>(std::llround(
            ctx->scan_col_ts_spacing_ns * static_cast<double>(packet_first_col)));
        ctx->active_scan_stamp = packet_receive_time - rclcpp::Duration(0, offset_ns);
        ctx->stamp_ready = true;
    }

    auto scan_ts = first_nonzero_timestamp(ctx->active_scan->timestamp());
    if (scan_ts == 0) {
        scan_ts = static_cast<uint64_t>(packet_receive_time.nanoseconds());
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Profile '%s' using packet receive time for scan timestamp",
            ctx->profile.name.c_str());
    }

    if (ctx->profile.timestamp_mode == "TIME_FROM_PTP_1588") {
        scan_ts = impl::ts_safe_offset_add(scan_ts, ctx->ptp_utc_tai_offset_ns);
    }
    ctx->active_scan_ts = scan_ts;

    if (!is_ros_time_mode(ctx->profile.timestamp_mode) || !ctx->stamp_ready) {
        ctx->active_scan_stamp = rclcpp::Time(scan_ts, node_clock_type());
        ctx->stamp_ready = true;
    }
    ctx->expect_frame_start = false;
}

void OusterDriver::handle_streaming_packet(
    const std::shared_ptr<SensorContext>& ctx,
    const sensor::LidarPacket& packet) {
    if (!ctx->scan_batcher || !ctx->active_scan || !ctx->packet_format) return;

    const auto packet_receive_time =
        rclcpp::Time(packet.host_timestamp, node_clock_type());
    const int packet_first_col =
        packet_col_index(*ctx->packet_format, packet.buf.data());

    std::lock_guard<std::mutex> lock(ctx->scan_mutex);
    bool scan_complete = (*ctx->scan_batcher)(packet, *ctx->active_scan);
    update_scan_timestamps_locked(ctx, packet_receive_time, packet_first_col);
    ctx->scan_dirty.store(true, std::memory_order_release);
    if (scan_complete) {
        ctx->expect_frame_start = true;
    }
}

bool OusterDriver::snapshot_active_scan(
    const std::shared_ptr<SensorContext>& ctx, ouster::LidarScan& scan_copy,
    uint64_t& scan_ts, rclcpp::Time& msg_ts) {
    if (!ctx->scan_dirty.exchange(false)) return false;

    std::lock_guard<std::mutex> lock(ctx->scan_mutex);
    if (!ctx->active_scan || !ctx->stamp_ready || ctx->active_scan_ts == 0)
        return false;

    scan_copy = *ctx->active_scan;
    scan_ts = ctx->active_scan_ts;
    msg_ts = ctx->active_scan_stamp;

    if (ctx->min_scan_valid_columns_ratio > 0.0) {
        auto status = scan_copy.status();
        const size_t valid_cols =
            std::count_if(status.data(), status.data() + status.size(),
                          [](const uint32_t s) { return (s & 0x01); });
        const auto min_required =
            static_cast<size_t>(ctx->min_scan_valid_columns_ratio *
                                static_cast<double>(status.size()));
        if (valid_cols < min_required) {
            return false;
        }
    }

    return true;
}

bool OusterDriver::transform_cloud(const sensor_msgs::msg::PointCloud2& in,
                                   sensor_msgs::msg::PointCloud2& out,
                                   const std::string& frame,
                                   const rclcpp::Time& stamp) {
    try {
        auto tf = tf_buffer_->lookupTransform(
            fusion_frame_, frame, stamp,
            tf2::durationFromSec(fusion_tf_timeout_));
        tf2::doTransform(in, out, tf);
        out.header.frame_id = fusion_frame_;
        out.header.stamp = stamp;
        return true;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "TF lookup failed: %s", ex.what());
        return false;
    }
}

sensor_msgs::msg::PointCloud2 OusterDriver::fuse_transformed_clouds(
    const std::vector<sensor_msgs::msg::PointCloud2>& clouds,
    const std::string& frame, const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 fused;
    if (clouds.empty()) {
        fused.header.frame_id = frame;
        fused.header.stamp = stamp;
        return fused;
    }

    sensor_msgs::msg::PointCloud2 current = clouds.front();
    for (size_t i = 1; i < clouds.size(); ++i) {
        sensor_msgs::msg::PointCloud2 tmp;
        pcl::concatenatePointCloud(current, clouds[i], tmp);
        current = tmp;
    }

    fused = current;
    fused.header.frame_id = frame;
    fused.header.stamp = stamp;
    return fused;
}

bool OusterDriver::collect_transformed_clouds(
    const std::shared_ptr<SensorContext>& ctx,
    std::vector<sensor_msgs::msg::PointCloud2>& transformed,
    rclcpp::Time& msg_ts) {
    if (!ctx->lidar_scan_processor) return false;

    ouster::LidarScan scan_snapshot{
        ctx->info.format.columns_per_frame, ctx->info.format.pixels_per_column,
        ctx->info.format.udp_profile_lidar};
    uint64_t scan_ts = 0;
    if (!snapshot_active_scan(ctx, scan_snapshot, scan_ts, msg_ts)) {
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Profile '%s' no snapshot yet (dirty=%d, stamp_ready=%d, ts=%lu)",
            ctx->profile.name.c_str(),
            ctx->scan_dirty.load(std::memory_order_relaxed) ? 1 : 0,
            ctx->stamp_ready ? 1 : 0, static_cast<unsigned long>(scan_ts));
        return false;
    }

    ctx->lidar_scan_processor(scan_snapshot, scan_ts, msg_ts);

    std::vector<sensor_msgs::msg::PointCloud2> local_clouds;
    {
        std::lock_guard<std::mutex> lock(ctx->clouds_mutex);
        if (ctx->processed_clouds.empty()) return false;
        local_clouds = ctx->processed_clouds;
    }

    transformed.clear();
    transformed.reserve(local_clouds.size());
    for (const auto& cloud : local_clouds) {
        sensor_msgs::msg::PointCloud2 tf_cloud;
        if (!transform_cloud(cloud, tf_cloud, cloud.header.frame_id, msg_ts))
            continue;
        transformed.push_back(std::move(tf_cloud));
    }
    return !transformed.empty();
}

void OusterDriver::fusion_timer_callback() {
    if (sensor_contexts_.empty()) return;

    using clock = std::chrono::steady_clock;
    auto callback_start = clock::now();
    struct SensorTiming {
        std::string name;
        double ms;
        size_t clouds;
    };
    std::vector<SensorTiming> sensor_timings;

    std::vector<std::vector<sensor_msgs::msg::PointCloud2>> per_return;
    per_return.resize(fusion_pubs_.size());

    bool have_data = false;
    rclcpp::Time fused_stamp = get_clock()->now();

    for (const auto& ctx : sensor_contexts_) {
        auto sensor_start = clock::now();
        std::vector<sensor_msgs::msg::PointCloud2> transformed;
        rclcpp::Time msg_ts;
        if (!collect_transformed_clouds(ctx, transformed, msg_ts)) {
            RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                                  "Profile '%s' has no new cloud data yet",
                                  ctx->profile.name.c_str());
            continue;
        }
        have_data = true;
        if (msg_ts < fused_stamp) fused_stamp = msg_ts;
        for (size_t i = 0; i < transformed.size() && i < per_return.size();
             ++i) {
            per_return[i].push_back(transformed[i]);
        }
        auto sensor_end = clock::now();
        double sensor_ms =
            std::chrono::duration<double, std::milli>(sensor_end - sensor_start)
                .count();
        sensor_timings.push_back(
            SensorTiming{ctx->profile.name, sensor_ms, transformed.size()});
    }

    if (!have_data) {
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                              "Fusion timer waiting for sensor data");
        return;
    }

    std::vector<double> build_timings(fusion_pubs_.size(), 0.0);
    std::vector<size_t> build_cloud_counts(fusion_pubs_.size(), 0);
    double publish_ms_total = 0.0;

    for (size_t i = 0; i < fusion_pubs_.size(); ++i) {
        auto build_start = clock::now();
        auto fused_msg =
            fuse_transformed_clouds(per_return[i], fusion_frame_, fused_stamp);
        build_timings[i] =
            std::chrono::duration<double, std::milli>(clock::now() - build_start)
                .count();
        build_cloud_counts[i] = per_return[i].size();
        auto publish_start = clock::now();
        fusion_pubs_[i]->publish(fused_msg);
        publish_ms_total +=
            std::chrono::duration<double, std::milli>(clock::now() - publish_start)
                .count();
    }

    auto total_ms =
        std::chrono::duration<double, std::milli>(clock::now() - callback_start)
            .count();

    for (const auto& timing : sensor_timings) {
        RCLCPP_DEBUG(get_logger(),
                     "[fusion] sensor '%s' processed %zu returns in %.3f ms",
                     timing.name.c_str(), timing.clouds, timing.ms);
    }
    for (size_t i = 0; i < build_timings.size(); ++i) {
        RCLCPP_DEBUG(get_logger(),
                     "[fusion] fused return %zu from %zu clouds in %.3f ms",
                     i, build_cloud_counts[i], build_timings[i]);
    }
    RCLCPP_DEBUG(get_logger(),
                 "[fusion] published %zu returns in %.3f ms "
                 "(publish %.3f ms total)",
                 fusion_pubs_.size(), total_ms, publish_ms_total);
}

}  // namespace ouster_ros

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(ouster_ros::OusterDriver)
