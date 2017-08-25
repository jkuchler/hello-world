# Tensorflow - IBM PowerAI Distributed Deep Learning (DDL) Integration

Tensorflow version: 1.1.0  
Machines: S822LC with NVLINK-attached P100 GPU (ppc64el, Ubuntu 16.04)  

## Summary for the Impatient

We propose to run TensorFlow distributed over multiple nodes, using multiple
GPUs. We do not intend to use TensorFlow's native capabilities in doing so.
We use MPI to kick off multiple TensorFlow processes, as many on a node as
directed by some rankfile, typically one process per available GPU.
All the TensorFlow processes run independently; they do not know of each
other's existence. The result of this is for one that any console messages (to
stdout or stderr) are all faithfully redirected by MPI to a single console.
So be not surprised of many replicated messages. Other than the IBM
PowerAI Distributed Deep Learning (DDL) primitives, there is no other means
of communication/synchronization.

We use the TF-Slim scripts.
This means that during training there will be no testing/validation phase; a
separate script has to be run after the training on the snapshot model.
We use synchronous weight updates. This means that the average gradient is
shared among the TensorFlow instances. The slowest process will determine the
duration of an iteration. Moreover, gradient averaging and sharing happens
only after all gradients per process are computed.
There is no (or perhasp very little) overlap of computation and communication.

## Introduction

TensorFlow is an open-source software library for Machine Intelligence.
Its core algorithms are implemented in C++. It also has an extensive Python
API. On top of the Python API several third-party software packages are
offered, like Slim and Keras.

Slim is very popular for Deep Learning applications. It offers a catalog of
ready-made neural networks and a generic learning/inference script.

IBM PowerAI Distributed Deep Learning (DDL) is an IBM proprietary
software library that supports communication and synchronization among
multiple hosts CPUs and connected GPU devices.  It builds on Open MPI
and NVIDIA's NCCL. DDL extends this library with several TensorFlow operators.

## Goal

The objective is to integrate Distributed Deep Learning in TensorFlow to
facilitate scale-up of large Deep Learning problems across multiple nodes and
many GPUs. In particular we want TensorFlow to run in single node, single GPU
mode, and have DDL control the deployment of several TensorFlow instances.
Although TensorFlow purportedly has some machinery available to do its own
cloning across GPUs and CPUs we want to make sure that these facilities are
disabled.

## Approach

TensorFlow lets a user define a computation (data-flow) graph which
subsequently is executed for purposes of for instance training a neural
network. The graph consists of operator nodes that perform a certain operation
on incoming tensors to produce output result tensors. There are many
built-in operator nodes both for mathematical operations but also for
logical operations and control-flow. A user can also define her own operators
either at the Python level or directly in C++.
TensorFlow takes the use of the computation graph to the extreme: everything
needed for a particular application will be part of the graph.

![Operator Graph](images/cifar10_overview.png "")
*Figure 1:* TensorBoard screenshot of the complete CifarNet graph.

Figure 1 shows a screenshot of the accompanying TensorBoard program that
offers a web browser based graphical user-interface. The "Graphs" tab presents
a drawing of the operator graph, in this case for the CifarNet convolution
network. The greyed-out rectangle is expanded on the right and shows the
neural net structure; in it, we zoom in on the first convolution and show
details of its weights.

From a Deep Learning perspective, the TensorFlow graph will consist at least of
the neural network under consideration. There will be graph nodes for
operations like convolution, pooling, inner-product, etc. Although at the Slim
level many details are left implicit, the graph eventually will explicitly
have nodes for weight and bias variables and even nodes for their one-time
initialization. In contrast, Caffe will keep the layer parameters implicit in
its graph representation; they are also not visible in the Caffe network
prototxt file. The consequence is that TensorFlow graph files can become very
large (`inception-v3.pbtxt`, a textual protobuffer file, is 19MB; the Caffe
equivalent is a mere 100kB).

A good overview of TensorFlow's architecture can be found in
[whitepaper2015 (pdf)][1].
The [documentation][5] on the internet is helpful as well.
It is less clear how the Slim layer on top of TensorFlow initiates a
multi-clone run and what the notion of replicas precisely means.
It is a fact that TensorFlow will enlist all the GPUs it detects on a
particular node and sets up a Cuda context for each, even though it does not
use all of them. This has the effect that some 287MB of memory is allocated on
all GPUs in the case where we constrain TensorFlow to grow the memory instead
of grabbing all of it (`gpu_options.allow_growth = True`).

This will also be the case when TensorFlow is run as part of Distributed
Deep Learning: there will be a TensorFlow instance per GPU and each will
pre-allocate some memory for all GPUs it detects. Ergo, the amount of wasted
memory is multiplied by the number of GPUs we use on a node. This also explains
why we see multiple python processes per GPU in the output of `nvidia-smi`.
For the case of 4 GPUs the DDL approach will waste 1GB of GPU memory.

The following considerations lead to a solution of successfully integrating
DDL with TensorFlow:

1. No change to the core TensorFlow code; neither Python nor C++ is touched.
2. Any changes should be as unobtrusive to a user as possible.
3. We prefer to use the Slim package and scripts: they offer many interesting
   models that are deployed through a single parametrized training script.

Since the Slim training script is the entry point of running TensorFlow,
we decided to modify it such that appropriate calls are made to the DDL
API functions. The necessary modifications can easily be applied via a
so-called patch file.
Integration involves several steps:

1. Making sure the DDL subsystem gets properly initialized.
2. Intercepting the initialization of all trainable variables, i.e., all
   weights and biases.
3. Making sure that the weights (and biases) are kept in-sync across
   TensorFlow instances.
4. Distributing the input data across the multiple TensorFlow instances.
5. Ensuring that the major computations are mapped to a particular GPU device
   as assigned by DDL.

Note that initialization of DDL is independent of the TensorFlow
computation graph. Since it returns the Open MPI rank number and an assigned
GPU device id it has to be done before input data streams are set up and the
computation graph is constructed. The (Python) function we use is called
`ddl_MDR.init()`.

Weight and bias variables are (memory) allocated and initialized as the result
of execution of a subgraph of variable, assignment, and (often random) initial
value nodes.
We want DDL to learn the initial value (tensor) of each variable and
have it interject its own value in its place. The intention is that all
instances of a given variable will be initialized with the exact same value as
broadcast by DDL. Mind that DDL is not a separate entity here
but comprises the notion of the distributed control for
synchronization and communication. Each variable will be initialized via a
graph node that is created by a call to `ddl_MDR.bcast()`. That node when
executed will in turn call e.g. MPI_Bcast() to effectuate the actual
collective broadcast operation. Clearly we must ensure that the broadcast
nodes are evaluated by TensorFlow in the same order for each instance.
For the longest time (on various platforms) this indeed seemed to be the case
without any further explicit ordering control edges. However, recently on RHEL
7.3 ppc64le problems with Bcast crashing have been reported.
A quick naive fix would be to remove the introduction of Bcast operator nodes
and rely on enforcing the same seed to the random number generator of each
TensorFlow instance. Hopefully this causes the corresponding weight and bias
variables of the various instances to be initialized to the exact same values
which is precisely what the Bcasts were intended for. Things can still go
wrong if weights are initialized in different orders though.
The TensorFlow documentation states this:

   > "As the op added by tf.global_variables_initializer() initializes all variables in parallel..."

To fix this problem once and for all, explicit control dependency edges have
been introduced among all the Bcast nodes such that the nodes will have to be
executed serially in the order of variable creation.

We want the TensorFlow instances to train synchronously. This requires a
tight-coupling of the weight updates. The gradients computed by each instance
will be averaged and the result shared among the instances. TensorFlow has
specific nodes in the graph for these operations. Distributed Deep
Learning will intercept the gradients (tensors), do the averaging and send the
result back to the proper update nodes in each TensorFlow instance. This is a
slightly simpler and possibly less accurate approach than having one worker
update its weights and then share those weights among all workers. However, the
chosen approach avoids some pitfalls with respect to scaling of the learning
rate and control over momentum etc. We will use the `ddl_MDR.all_reduce()`
function to create the AllReduce graph nodes.

(When defining a new operator in C++, Tensorflow will automatically provide it
with a Python wrapper. The C++ operator name must be CamelCase; the Python
wrapper name will be generated all lowercase with inserted underscores.
Hence "AllReduce" becomes "all_reduce".)

TF-Slim has a sophisticated data input procedure. It can read data from a file
in `TFRecord` format and automatically shuffle the samples and queue them up
after preprocessing. Multiple threads can be used to simultaneously read from
several files. In a multi-instance setting, the data preferably is
partitioned and distributed among the instances. Tensorflow does this by
having reader(s) successively open files from a shuffled list of file names.
Most data sets consist of many files of samples (ImageNet has some 1000 files,
each of 1000 images). Without actually partitioning the data one has to
realize that after each instance processes its file list, we have in fact
processed N epochs, where N is the number of instances. For small models like
cifar10 Tensorflow does not seem to have a means to partition its single data
file. Trying to train cifar10 in a distributed way therefore does not make
much sense, at least not with the given script.

The deployment of the computation on a given GPU is enforced by using a
top-level `with tf.device('/gpu:%d' % gpuid):` statement where the gpuid value
is supplied by the DDL init call. Some nodes however are required to
run on a CPU, hence we allow Tensorflow to modify the assignment in those cases
by setting the configuration flag `allow_soft_placement=True`.

## Gradients AllReduce

Sharing the gradients (to be precise, it is the average of the gradients that
must be computed and shared) among the instances is vital for a correct
operation. We must see to it that the same weights and biases across the
various instances are "connected" to the appropriate AllReduce graph node.
In a first implementation, AllReduce nodes were created and inserted in the
same order as the gradient and variable pairs in a particular list provided by
TensorFlow. That list is not necessarily ordered in anyway according to the
TensorFlow documentation and might even be different across instances.

It should be understood that the Distributed Deep Learning `bcast()` and
`all_reduce()` operators align
themselves across instances by their implicit order of creation. There is no
explicit identification of these nodes. If this alignment
somehow fails, i.e., it is not respected by the run-time execution of the
graph nodes, the effect will be that an AllReduce is attempted on probably
differently sized tensors which most likely will lead to a segmentation
violation. On top of that there is the problem of non-deterministic choices
made by the TensorFlow graph execution scheduler, maybe caused by differences
in execution speed of some kernels. This is very likely to happen in deep
learning networks that have many branching layers such as inception and resnet
models. Indeed in our first naive implementation we encounter this problem
and end up with a dead-locked ("hanging") system. One solution is to enforce
an arbitrary sequential execution of the AllReduce nodes after it is ensured
that all gradient tensors are available. This is implemented with the use of
a `with tf.control_dependencies():` statement. Mind that TensorFlow does not
allow additional control dependency edges to be added between nodes after they
have been created; nodes are to be considered immutable.

An alternative solution has evolved consisting of using just a single
AllReduceN node that accepts a list of gradient tensors. The benefits are of
course simplicity of use, guaranteed correct scheduling by TensorFlow, and
the opportunity for performance improvements since all gradients are known to
DDL at the same time.

## Determinism

It is nigh impossible to have reproducible Tensorflow experiments.
For some as yet unknown reason Tensorflow seems to use random numbers
beyond the control of a user. One would think that a simple single GPU
run would render (almost) identical loss values across runs. We see
deviations up to 10% (differences in 2nd decimal after decimal point).
Some obvious sources of variations are the seeds used by the various random
number generators. Python has its own, so does numpy and so does Tensorflow.
The latter can be seeded with the function `tf.set_random_seed()` which
has to be called within the context of a computation graph.

Even with all these random seeds fixed, it is well possible to observe large
fluctuations in numerical results simply because of the massively parallel
nature of the GPU execution. On a CPU, one should be aware of the fact that
the Eigen linear algebra library uses OpenMP to parallelize its operations.
Switching the number of Eigen threads to 1 (`intra_op_parallelism_threads=1`)
clearly shows a more reproducible result albeit at a speed loss of a factor 10.

Another (significant) source of variations is the data reader part. The data
shuffler uses
its own random seed which can be controlled as argument to the
`DatasetDataProvider` constructor. Moreover, the reader typically uses multiple
threads both for reading files as well as for performing any preprocessing.
The readers write to queues in an asynchronous fashion. There is no guarantee
of
a certain order of the input data samples in the queue when multiple readers
enqueue the data. To attain reproducible behavior one must use a single reader
by setting the options `--num_readers=1` and `--num_preprocessing_threads=1`.

On top of all this, image preprocessing obviously might add yet another source
of non-determinism: typically images will be padded then randomly cropped,
randomly flip and distorted. To avoid this, preprocessing is best disabled
all together by commenting out the call to `image_preprocessing_fn()`.

See also: [stackoverflow: determinism-in-tensorflow][4].

## Current Status

The `models/slim/train_image_classifier.py` has been modified with some
additional code to properly initialize DDL and insert its
communication operators in the Tensorflow Graph. The one missing aspect is
control over the data input. Ideally, the dataset should be partitioned across
the TensorFlow instances. This would require some filter operator that
depending on the MPI rank and the number of instances N, selects 1 out
of every N samples. For now we rely on the fact that most large datasets are
provided by a large number of files and these files are shuffled per instance.

## TensorFlow-Slim Native Distribution

Slim provides the `train_image_classification.py` script that offers the
`--num_clones` option. Using this option set to the value 2 for instance,
causes the model sub-graph to be cloned (or maybe better replicated).
Analysis of the graph in TensorBoard will show a clone_0 sub-graph and a
clone_1 sub-graph each containing the model graph under consideration
(e.g. cifarnet). clone_0 would be mapped onto GPU0 and clone_1 on to GPU1.
The parameters (weights and biases) are shared and mapped to the CPU.

![CifarNet 2 clones](images/clones2.png "2 clones of CifarNet")
*Figure 2:* TensorBoard screenshot showing an Operator Graph with 2 clones of
CifarNet.

## Practical Considerations

Here we discuss how to run TensorFlow across multiple machines and use multiple
GPUs on each host. To run on multiple machines or even on a single machine but
multiple GPUs, the MPI framework is deployed to start up TensorFlow processes.
Across multiple hosts, MPI will take care of setting up communication
connections to the various hosts and starting the required number of
TensorFlow processes. There will be precisely 1 TensorFlow process per GPU;
we do not intend to ever run TensorFlow on a CPU.

We assume TensorFlow has been installed (on all the target machines)
preferably in a virtual Python environment by pip installing the appropriate
TensorFlow wheel package. Typically this would result in `/opt/DL/tensorflow`.
The MPI utility program `mpirun` can be used to spawn a user program across
multiple machines as specified by certain command-line arguments or a
so-called rank file.

A user needs to establish a proper shell environment (`PATH` and
`LD_LIBRARY_PATH`) by sourcing `/opt/DL/tensorflow/bin/tensorflow-activate`.
These environment variables must be passed on via `mpirun`'s `-x` options to
the target machines. Any other means of ensuring that the processes on the
target hosts are run in a proper environment (where they can find the required
libraries and other dependencies, e.g. `PYTHONPATH`) is fine too.

TensorFlow is invoked via a TF-Slim Python script, namely
`train_image_classifier.py` to be found in the directory `models/slim`.
Mind that Slim in not part of the TensorFlow install and needs to be
downloaded or git-cloned from `https://github.com/tensorflow/models`.
The original `train_image_classifier.py` script must be replaced by the one
provided by the DDL package.

The `train_image_classifier.py` script is best run from the `models/slim`
directory where it resides:

```bash
$ cd models/slim
$ python train_image_classifier.py [slim-options]
```

This is a correct call for the original script. The modified script needs to
be called under control of mpirun:

```bash
$ cd models/slim
$ mpirun [mpirun-options] python train_image_classifier.py [slim-options]
```

## References

Some additional sources that are useful to peruse are:

1. [User-transparent Distributed TensorFlow (pdf)][2]

2. [Distributed TensorFlow with MPI (pdf)][3]


[1]: http://download.tensorflow.org/paper/whitepaper2015.pdf
"TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems"

[2]: https://arxiv.org/pdf/1704.04560.pdf "User-transparent Distributed TensorFlow"

[3]: https://arxiv.org/pdf/1603.02339.pdf "Distributed TensorFlow with MPI"

[4]: http://stackoverflow.com/questions/39938307/determinism-in-tensorflow-gradient-updates "stackoverflow: determinism-in-tensorflow"

[5]: https://www.tensorflow.org/extend/architecture "TensorFlow Architecture"
