#!/usr/bin/env python3
"""
测试Blitting优化效果
对比优化前后的性能差异
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def test_original_approach():
    """测试原始方法（每次清除重绘）"""
    print("\n=== 测试原始方法（每次清除重绘）===")
    
    fig = plt.Figure(figsize=(10, 8), dpi=100)
    axs = []
    
    for i in range(8):
        ax = fig.add_subplot(9, 1, i+1)
        ax.set_ylabel(f'EMG {i+1}')
        ax.set_ylim(-50000, 50000)
        ax.grid(True, alpha=0.3)
        axs.append(ax)
    
    ax = fig.add_subplot(9, 1, 9)
    ax.set_ylabel('IMU')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    axs.append(ax)
    
    canvas = FigureCanvasTkAgg(fig, master=None)
    canvas.draw()
    
    # 模拟数据
    times = list(range(100))
    emg_data = np.random.randn(100, 8) * 10000
    imu_data = np.random.randn(6) * 2
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
    
    # 测试性能
    iterations = 100
    start_time = time.time()
    
    for _ in range(iterations):
        for ax in axs:
            ax.clear()
        
        for i in range(8):
            axs[i].plot(times, emg_data[:, i], color=colors[i], linewidth=1)
            axs[i].set_ylabel(f'EMG {i+1}')
            axs[i].set_ylim(-50000, 50000)
            axs[i].grid(True, alpha=0.3)
        
        axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], imu_data)
        axs[8].set_ylabel('IMU')
        axs[8].set_ylim(-5, 5)
        axs[8].grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas.draw()
    
    end_time = time.time()
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    
    print(f"总时间: {elapsed:.3f}秒")
    print(f"平均每次更新: {avg_time*1000:.2f}毫秒")
    print(f"FPS: {1/avg_time:.1f}")
    
    plt.close(fig)
    
    return avg_time


def test_blitting_approach():
    """测试Blitting方法"""
    print("\n=== 测试Blitting方法 ===")
    
    fig = plt.Figure(figsize=(10, 8), dpi=100)
    axs = []
    
    for i in range(8):
        ax = fig.add_subplot(9, 1, i+1)
        ax.set_ylabel(f'EMG {i+1}')
        ax.set_ylim(-50000, 50000)
        ax.grid(True, alpha=0.3)
        axs.append(ax)
    
    ax = fig.add_subplot(9, 1, 9)
    ax.set_ylabel('IMU')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    axs.append(ax)
    
    canvas = FigureCanvasTkAgg(fig, master=None)
    
    # 初始化线条对象
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
    lines = []
    for i in range(8):
        line, = axs[i].plot([], [], color=colors[i], linewidth=1)
        lines.append(line)
    
    # 初始化柱状图
    bar_container = axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], [0]*6)
    
    plt.tight_layout()
    
    # 保存背景
    canvas.draw()
    backgrounds = [canvas.copy_from_bbox(ax.bbox) for ax in axs]
    
    # 模拟数据
    times = list(range(100))
    emg_data = np.random.randn(100, 8) * 10000
    imu_data = np.random.randn(6) * 2
    
    # 测试性能
    iterations = 100
    start_time = time.time()
    
    for _ in range(iterations):
        for i in range(8):
            # 恢复背景
            canvas.restore_region(backgrounds[i])
            
            # 更新数据
            lines[i].set_data(times, emg_data[:, i])
            
            # 只重绘线条
            axs[i].draw_artist(lines[i])
            
            # blit到画布
            canvas.blit(axs[i].bbox)
        
        # 更新IMU
        canvas.restore_region(backgrounds[8])
        for rect, h in zip(bar_container, imu_data):
            rect.set_height(h)
        
        for rect in bar_container:
            axs[8].draw_artist(rect)
        
        canvas.blit(axs[8].bbox)
    
    end_time = time.time()
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    
    print(f"总时间: {elapsed:.3f}秒")
    print(f"平均每次更新: {avg_time*1000:.2f}毫秒")
    print(f"FPS: {1/avg_time:.1f}")
    
    plt.close(fig)
    
    return avg_time


def main():
    """主函数"""
    print("=" * 60)
    print("Blitting优化性能测试")
    print("=" * 60)
    
    # 测试原始方法
    original_time = test_original_approach()
    
    # 测试Blitting方法
    blitting_time = test_blitting_approach()
    
    # 计算性能提升
    improvement = (original_time - blitting_time) / original_time * 100
    speedup = original_time / blitting_time
    
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"原始方法平均更新时间: {original_time*1000:.2f}毫秒")
    print(f"Blitting方法平均更新时间: {blitting_time*1000:.2f}毫秒")
    print(f"性能提升: {improvement:.1f}%")
    print(f"速度提升: {speedup:.1f}x")
    print("=" * 60)
    
    if improvement > 50:
        print("\n✓ Blitting优化效果显著！")
    elif improvement > 20:
        print("\n✓ Blitting优化效果良好！")
    else:
        print("\n⚠ Blitting优化效果一般，可能需要进一步调整")


if __name__ == "__main__":
    main()