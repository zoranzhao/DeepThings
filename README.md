<p align="center"><img src="https://zoranzhao.github.io/deepthings.png" alt="DeepThings Overview" title="DeepThings" width="200px"/></p>

# DeepThings
A C Library for Distributed CNN Inference on IoT Edge Clusters 


## Building

For most users, the recommended way to build NNPACK is through CMake:

```bash
./deepthings -mode start
./deepthings -mode gateway -total_edge 6 -n 5 -m 5 -l 16
./deepthings -mode data_src -edge_id 0 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 1 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 2 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 3 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 4 -n 5 -m 5 -l 16
./deepthings -mode non_data_src -edge_id 5 -n 5 -m 5 -l 16

./deepthings -mode <execution mode: {start, gateway, data_src, non_data_src}> 
             -total_edge <total edge number: t> 
             -edge_id <edge device ID: e={0, ... t-1}>
             -n <FTP dimension: N> 
             -m <FTP dimension: M> 
             -l <numder of fused layers: L>
```
