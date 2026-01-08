## cycleAccurateSimulators

This folder builds and runs the `accel_sim` cycle-accurate-ish accelerator simulator in `accel_sim.cpp`.

### Build

Using CMake (recommended):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or with the provided Makefile wrapper:

```bash
make build
```

### Run

```bash
./build/accel_sim
```

You can override parameters via CLI flags, for example:

```bash
./build/accel_sim --num_cus=8 --num_dram_channels=8 --dram_latency=300 --noc_bpc=64
```


