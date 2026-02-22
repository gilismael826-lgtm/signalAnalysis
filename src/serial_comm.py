#!/usr/bin/env python3
"""
串口通信模块
负责串口连接和数据接收
"""

import time
import src.config as config
from src.config import SERIAL_PORT, BAUDRATE, running, ser, emg_buffer, imu_buffer, logger
from src.config import PREPROCESSING_ENABLED
from src.data_parser import parse_packet
from src.data_manager import save_data_point
from src.signal_processor import signal_processor


def serial_reader():
    """串口数据接收线程"""
    global ser
    
    try:
        import serial
        
        # 确保 running 为 True
        logger.info(f"开始初始化串口: {config.SERIAL_PORT}, 波特率: {config.BAUDRATE}")
        
        # 尝试多次连接串口
        for attempt in range(3):
            try:
                ser = serial.Serial(config.SERIAL_PORT, config.BAUDRATE, timeout=1)
                logger.info(f"串口连接成功 (尝试 {attempt+1}/3): {config.SERIAL_PORT}")
                break
            except Exception as e:
                logger.warning(f"串口连接失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(0.5)
        else:
            logger.error("串口连接失败，已达到最大尝试次数")
            return
        
        # 等待串口稳定
        time.sleep(0.2)
        logger.info("串口初始化完成，开始接收数据")
        
        buffer = b''
        packet_count = 0
        error_count = 0
        last_activity_time = time.time()
        
        while config.running:
            try:
                # 检查串口是否仍然打开
                if not ser or not ser.is_open:
                    logger.error("串口已关闭，重新连接...")
                    ser = serial.Serial(config.SERIAL_PORT, config.BAUDRATE, timeout=1)
                    logger.info("串口重新连接成功")
                
                data = ser.read(1000)
                current_time = time.time()
                
                if data:
                    last_activity_time = current_time
                    logger.info(f"接收到 {len(data)} 字节数据")
                    buffer += data
                    
                    # 处理缓冲区中的数据包
                    while len(buffer) >= 29:
                        start = buffer.find(b'\xd2\xd2\xd2')
                        if start == -1:
                            logger.warning(f"未找到数据包起始标志，当前缓冲区长度: {len(buffer)}")
                            # 只保留最后一部分缓冲区，避免内存占用过大
                            if len(buffer) > 1000:
                                buffer = buffer[-1000:]
                            error_count += 1
                            break
                        
                        if start + 29 <= len(buffer):
                            packet = buffer[start:start+29]
                            buffer = buffer[start+29:]
                            
                            pkt_type, ts, parsed_data = parse_packet(packet)
                            if pkt_type:
                                packet_count += 1
                                if packet_count % 10 == 0:
                                    logger.info(f"已解析 {packet_count} 个数据包")
                                
                                if pkt_type == 'EMG':
                                    config.emg_buffer.append((ts, parsed_data))
                                    if config.is_collecting:
                                        config.collection_emg_buffer.append((ts, parsed_data))
                                    save_data_point('EMG', ts, parsed_data)
                                    logger.info(f"解析到EMG数据: {parsed_data[:2]}...")
                                    
                                    if config.PREPROCESSING_ENABLED:
                                        signal_processor.update_buffers(parsed_data, [])
                                elif pkt_type == 'IMU':
                                    config.imu_buffer.append((ts, parsed_data))
                                    if config.is_collecting:
                                        config.collection_imu_buffer.append((ts, parsed_data))
                                    save_data_point('IMU', ts, parsed_data)
                                    logger.info(f"解析到IMU数据: {parsed_data[:2]}...")
                                    
                                    # 预处理
                                    if config.PREPROCESSING_ENABLED:
                                        signal_processor.update_buffers([], parsed_data)
                            else:
                                error_count += 1
                                if error_count % 10 == 0:
                                    logger.warning(f"解析错误计数: {error_count}")
                
                # 检查是否超时（5秒无活动）
                if current_time - last_activity_time > 5:
                    logger.warning("串口无活动超过5秒")
                    # 发送一个小的请求数据包（如果设备支持）
                    try:
                        ser.write(b'\x00')
                    except:
                        pass
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"读取串口数据失败: {e}")
                # 发生异常时，短暂休眠后继续尝试
                time.sleep(0.1)
        
        # 循环结束，关闭串口
        if ser and ser.is_open:
            ser.close()
            logger.info("串口已关闭")
        
    except Exception as e:
        logger.error(f"串口线程异常: {e}")
        # 确保串口被关闭
        if ser and ser.is_open:
            try:
                ser.close()
            except:
                pass
