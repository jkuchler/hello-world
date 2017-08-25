# Overview

IBM PowerAI Distributed Deep Learning (or DDL) is a MPI-based
communication library, which is specifically optimized for Deep Learning
training.  An application integrated with DDL becomes a MPI-application,
which will allow the use of the `mpirun` command to invoke the job in
parallel across a cluster of systems. DDL understands multi-tier network
environment and uses different libraries (e.g. NCCL) and algorithms to
get the best performance in multi-node, multi-GPU environments.

IBM PowerAI Distributed Deep Learning considers each GPU in a cluster as
an individual "learner".  The overall set of learners is described to
IBM PowerAI Distributed Deep Learning in terms of 3 dimensions (X-Y-Z)
that correspond to a multi-tier network hierarchy.  The recommended
mapping is:

   - X for within-host (e.g. number of GPUs per host for multi-GPU hosts)
   - Y for between nearby-hosts (e.g. number of hosts in a single rack)
   - Z for between distant-hosts (e.g. number of racks)

For example, 256 learners can be configured as 4x8x8 or 4x16x4 and so on.

**Example: 2 racks of 8 S822LC for HPC systems with 4 GPUs each**

In this setup, there are 64 learners (4 GPUs in each of 8 hosts in each
of 2 racks) and a simple description would be 4x8x2.

If this configuration includes a truly hierarchical network setup--for
example a high-speed, low-latency network within each rack, and a
slower, higher-latency network between the racks--then 4x8x2 might be
the optimal description.

But if the network configuration is not actually hierarchical--if all
the hosts are connected to a "flat" network regardless of the physical
racking--then a 4x4x4 description may perform better than 4x8x2. Some
experimentation may be needed to find the optimal description.


# Required Libraries

Pre-requisite packages required for IBM PowerAI Distributed Deep
Learning are provided with PowerAI:

   1. OpenMPI with CUDA Support
   2. NVIDIA NCCL


# Integration with Caffe and TensorFlow

IBM PowerAI Distributed Deep Learning has been integrated with the
PowerAI IBM Caffe and TensorFlow packages. `mpirun` must be used to
launch training using the IBM PowerAI Distributed Deep Learning
integration.  General information about `mpirun` is available on the
OpenMPI website
[https://www.open-mpi.org/doc/v2.0/man1/mpirun.1.php](https://www.open-mpi.org/doc/v2.0/man1/mpirun.1.php).

   1. Caffe

      IBM PowerAI Distributed Deep Learning is directly integrated into
      Caffe, and can be exercised by adding the following to the command line.

           -ddl “DDL_OPTIONS HERE”

   2. TensorFlow

      DDL is indirectly integrated into TensorFlow in the form of custom
      operator. The custom operator is provided as a shared library, which is
      loaded and invoked in the python training script.

      The PowerAI ddl-tensorflow package provides an example training
      setup based on the TensorFlow-Slim model library from the TensorFlow
      models repository. Those can be found on your system in:

           /opt/DL/ddl-tensorflow/examples/

      More details on IBM PowerAI Distributed Deep Learning integration
      into TensorFlow, can be found in

           /opt/DL/ddl-tensorflow/doc/README.md


# Using IBM PowerAI Distributed Deep Learning

IBM PowerAI Distributed Deep Learning takes advantage of the network
topology to perform communication quickly. Network topology is described
to IBM PowerAI Distributed Deep Learning in two ways, through an MPI
rank file and via DDL options.

## MPI rank file

A rank file is a standard file that maps MPI clients (IBM PowerAI
Distributed Deep Learning learners) to specific hosts, sockets, and
cores. To get the best performance from IBM PowerAI Distributed Deep
Learning , it is crucial to generate an optimally mapped rank file. To
help with this effort, a script (`rank_gen.py`) is provided to
automatically generate rank files that are appropriate for most S822LC
systems. The script takes two inputs: the network decomposition and a
list of comma-separated hostnames.

**How to use rank file generator script**

        $ python rank_gen.py XxYxZ host_list > rank_file

Here, `XxYxZ` specifies the topology of the GPU and multi-tier network
hierarchy.

For example, for 64 learners (e.g. 16 hosts each with 4 GPUs), any of
4x16x1, 4x8x2, or 4x4x4 might be reasonable choices, depending on the
network topology. All 3 dimensions must be specificed (use 1 to fill any
spaces).

`host_list` is a comma separated list of host names (e.g.
host1,host2,host3,...).  It must contain `Y` times `Z` hostnames,
ordered "by Z". For example, a 4x2x2 configuration with 2 racks of 2
hosts each might have a host list of: `r1h1,r1h2,r2h1,r2h2`.The
hostnames provided in the rankfile should match the system hostnames.

It is possible in a distributed environment to have more than one
interface for each host. In such a scenario, OpenMPI by default, uses
any and all interfaces that are "up" to communicate with a host. To
avoid problems in such cases you can tell OpenMPI to use given
interface. E.g.:

        $ mpirun --mca btl_tcp_if_include ib0 ...

        $ mpirun --mca btl_tcp_if_exclude lo,enp1s0f2 ...

More details available on OpenMPI FAQ page:
[https://www.open-mpi.org/faq/?category=tcp#tcp-selection]([https://www.open-mpi.org/faq/?category=tcp#tcp-selection)

**Parameters for optimal rankfile**

An optimal rank file will depend on the number of sockets or nodes in
the system and the number of cores per socket/node. The `numactl` and
`ppc64_cpu` commands can help determine this information.

   1. Number of sockets and thread slots for each socket.

      `numactl -H` shows the number of sockets ("nodes") in a system,
      and also lists the CPUs (thread slots) for each. For example:

           $ numactl -H
           available: 2 nodes (0-1)
           node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
           node 0 size: 261788 MB
           node 0 free: 6042 MB
           node 1 cpus: 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
           node 1 size: 261334 MB
           node 1 free: 158805 MB
           node distances:
           node   0   1
             0:  10  40
             1:  40  10

      Here the system has two sockets with 80 thread slots each.

   2. Mapping between physical cores and CPUs/thread slots.

           $ ppc64_cpu --info
           Core   0:    0*    1*    2*    3*    4*    5*    6*    7*
           Core   1:    8*    9*   10*   11*   12*   13*   14*   15*
           Core   2:   16*   17*   18*   19*   20*   21*   22*   23*
           Core   3:   24*   25*   26*   27*   28*   29*   30*   31*
           Core   4:   32*   33*   34*   35*   36*   37*   38*   39*
           Core   5:   40*   41*   42*   43*   44*   45*   46*   47*
           Core   6:   48*   49*   50*   51*   52*   53*   54*   55*
           Core   7:   56*   57*   58*   59*   60*   61*   62*   63*
           Core   8:   64*   65*   66*   67*   68*   69*   70*   71*
           Core   9:   72*   73*   74*   75*   76*   77*   78*   79*
           Core  10:   80*   81*   82*   83*   84*   85*   86*   87*
           Core  11:   88*   89*   90*   91*   92*   93*   94*   95*
           Core  12:   96*   97*   98*   99*  100*  101*  102*  103*
           Core  13:  104*  105*  106*  107*  108*  109*  110*  111*
           Core  14:  112*  113*  114*  115*  116*  117*  118*  119*
           Core  15:  120*  121*  122*  123*  124*  125*  126*  127*
           Core  16:  128*  129*  130*  131*  132*  133*  134*  135*
           Core  17:  136*  137*  138*  139*  140*  141*  142*  143*
           Core  18:  144*  145*  146*  147*  148*  149*  150*  151*
           Core  19:  152*  153*  154*  155*  156*  157*  158*  159*

      Here the system has 20 physical cores with 8 thread slots/CPUs each. The
      thread slot numbers match with the numbers in the `numactl` output. The
      asterisks indicate which thread slots are enabled.

      The rankfile only cares about cores (not CPUs/thread slots), and the
      core numbering is relative to the to the node/socket (which is named
      "slot" in the rankfile). So in rank file terms, this system has socket 0
      cores 0-9 and socket 1 cores 0-9.

**Note:** If the number of cores specified in the rankfile exceeds the
actual number of cores, `mpirun` will fail with a non-obvious message.
For example, on a machine with 8-cores per socket:

        $ cat 2x10core.rf
        rank 0=host1     slot=0:0-9
        rank 1=host1     slot=1:0-9

        $ mpirun -n 2 -rf 2x10core.rf /bin/true
        [host1:46256] [[20503,0],0] ORTE_ERROR_LOG: Not found in file rmaps_rank_file.c at line 320
        [host1:46256] [[20503,0],0] ORTE_ERROR_LOG: Not found in file base/rmaps_base_map_job.c at line 351
        $

Versus the working:

        $ cat 2x8core.rf
        rank 0=host1     slot=0:0-7
        rank 1=host1     slot=1:0-7

        $ mpirun -n 2 -rf 2x8core.rf /bin/true
        $

The `-report-bindings` flag may be useful for diagnosing problems:

        $ mpirun -report-bindings ......

## DDL options

There are a number of runtime options for the DDL engine. The options are:

`-mode`: This optionally indicates the algorithm and topology. The topology
should match to the rank assignment (e.g. via rankfile) to get the best
performance. If a mode is not provided, it will work as a single ring
configuration (e.g., r:N). Therefore, the total number of MPI clients
(specified as -n N to mpirun) must match with the number of learners in the
topology (specified as -mode in DDL): otherwise, it will show an error like
`invalid dim size=A usr_size=B dim[0]=...`

        b:4x2  => use enhanced NCCL whenever possible (otherwise use ring) for 4x2 configuration

        n:4x2  => use NCCL whenever possible (otherwise use ring) for 4x2 configuration

        r:4x4  => use only RING for 4x4 configuration

        m:4x6  => use only MPI reduce_scatter and all_gatherV for 4x6 configuration (currently disabled)

        c:4x8  => use only RCS for 4x8 configuration

        p:4x16x4 => first activate ex"p"location mode to get the best algorithms for each dimension of 4x16x4

`-dump_iter <N>`: This optionally makes DDL dump network performance on
every N iterations

`-dev_sync <0, 1, or 2>` : This optionally calls cudaDeviceSynchronize
to minimize jitter, default is 0 (which means off). With 1, it
invokes sync once in the beginning of communication. With 2, it invokes
sync twice in the beginning AND end of communication

 `-rebind_iter <N>`: This optionally monitors variation every N
iterations, and performs rebind if a leaner has been slow for the last 3
checks. Too small number will incur variation-check overhead, but too
big number will make training suffer from variation for long time

 `-dbg_level <0,1,2>`: 0 for no, 1 for mild, and 2 for detailed debug
messages

When `dump_iter` is given, you can see the following periodically where
you can find which learner has the maximum jitter and end to end DDL
elapsed time. Also, per dimension, it shows runtime breakdown along with
the selected algorithm for that particular dimension.

![Alt text](ddl_dump.png?raw=true "DDL dump")


**Example of 2 racks of 8 S822LC HPC systems with 4 GPUs on each host**

Generate an appropriate rank file:

        $ python rank_gen.py 4x8x2 host0,host1,host2,….,host15 > 4x8x2.rf

To start IBM Caffe with `mpirun`, specifying rank file and DDL options:

        $ source /opt/DL/caffe/bin/caffe-activate

        $ mpirun -x PATH -x LD_LIBRARY_PATH -n 16 -rf 4x8x2.rf caffe train -solver solver.prototxt -gpu 0 -bvlc -ddl "-mode b:4x8x2 -dump_iter 100"

To start TensorFlow with `mpirun` using custom operator for DDL:

   - Update `train_image_classifier.py` to specify DDL options during
     initialization:

        ddl.Init(4, mode =’-mode b:4x8x2 -dump_iter 100’)

   - Execute with `mpirun`:

        $ source /opt/DL/ddl-tensorflow/bin/ddl-tensorflow-activate

        $ mpirun -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -n 16 -rf 4x8.2.rf python train_image_classifier.py ...
