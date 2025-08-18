#!/usr/bin/env python3
"""
测试多进程功能的脚本
"""
import multiprocessing as mp
import time
import os
import sys

def test_worker_function(task_id, sleep_time=1):
    """测试工作函数"""
    print(f"进程 {os.getpid()} 开始处理任务 {task_id}")
    time.sleep(sleep_time)
    print(f"进程 {os.getpid()} 完成任务 {task_id}")
    return f"任务 {task_id} 完成"

def test_multiprocessing():
    """测试多进程功能"""
    print("测试多进程功能...")
    print(f"系统CPU核心数: {mp.cpu_count()}")
    
    # 测试任务列表
    tasks = list(range(10))
    
    # 单进程测试
    print("\n=== 单进程测试 ===")
    start_time = time.time()
    results_single = []
    for task in tasks:
        result = test_worker_function(task, 0.1)
        results_single.append(result)
    single_time = time.time() - start_time
    print(f"单进程耗时: {single_time:.2f}秒")
    
    # 多进程测试
    print("\n=== 多进程测试 ===")
    start_time = time.time()
    
    # 设置多进程启动方法（在macOS上避免问题）
    if sys.platform == 'darwin':
        mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=4) as pool:
        results_multi = pool.map(test_worker_function, tasks)
    
    multi_time = time.time() - start_time
    print(f"多进程耗时: {multi_time:.2f}秒")
    
    # 性能对比
    speedup = single_time / multi_time
    print(f"\n性能提升: {speedup:.2f}x")
    
    # 验证结果
    assert len(results_single) == len(results_multi)
    print("所有任务完成，结果验证通过！")

if __name__ == "__main__":
    test_multiprocessing() 