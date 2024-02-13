import torch
import time


def measure_gpu_throughput(model, inputs):
    batch_size = inputs.size(0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            output = model(inputs[i:i + batch_size])
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end)/1000
    # print(latency)
    # throughput = inputs.size(0) * batch_size / latency
    throughput = batch_size / latency
    return throughput


def measure_latency_cpu_usage(model, test_inputs):
    # process = psutil.Process()
    # cpu_start = process.cpu_percent()
    start = time.time()
    predictions = model(test_inputs[0:1, :, :, :])
    # print(test_inputs.shape)
    end = time.time()
    # print(start, end)
    # cpu_end = process.cpu_percent()
    latency = 1000 * (end - start) #convert to ms
    # cpu_usage = cpu_end - cpu_start
    return latency