import torch
import time


def measure_gpu_throughput(model, inputs):
    batch_size = inputs.size(0)
    # inputs = inputs.to('cuda')
    # model = model.to('cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            output = model(inputs[i:i + batch_size])
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end)/1000
    throughput = batch_size / latency
    return throughput


def measure_latency_cpu_usage(model, test_inputs):
    start = time.time()
    predictions = model(test_inputs[0:1, :, :, :])
    # print(test_inputs.shape)
    end = time.time()
    latency = 1000 * (end - start) #convert to ms
    return latency