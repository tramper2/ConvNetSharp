using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Diagnostics;

namespace ConvNetSharp
{
    [DataContract]
    public class ConvLayer : LayerBase, IDotProductLayer
    {
        private const int BlockSize = 1024;
        private double[] deviceBiasesWeights;
        private double[] deviceFilterWeights;
        private double[] deviceInputWeigths;
        private double[] deviceOutputWeights;
        private GPGPU gpu;
        private bool gpuAllocated;

        public ConvLayer(int width, int height, int filterCount)
        {
            this.GroupSize = 2;
            this.L1DecayMul = 0.0;
            this.L2DecayMul = 1.0;
            this.Stride = 1;
            this.Pad = 0;

            this.FilterCount = filterCount;
            this.Width = width;
            this.Height = height;
        }

        [DataMember]
        public Volume Biases { get; private set; }

        [DataMember]
        public List<Volume> Filters { get; private set; }

        [DataMember]
        public int FilterCount { get; private set; }

        [DataMember]
        public double L1DecayMul { get; set; }

        [DataMember]
        public double L2DecayMul { get; set; }

        [DataMember]
        public int Stride { get; set; }

        [DataMember]
        public int Pad { get; set; }

        [DataMember]
        public double BiasPref { get; set; }

        [DataMember]
        public Activation Activation { get; set; }

        [DataMember]
        public int GroupSize { get; private set; }

        private void InitGpu()
        {
            // optimized code by @mdda that achieves 2x speedup over previous version
            Console.WriteLine("InitGpu");

            try
            {
                CudafyModes.Target = eGPUType.Cuda; // To use OpenCL, change this enum
                CudafyModes.DeviceId = 0;
                CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

                var km = CudafyTranslator.Cudafy();
                this.gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
                this.gpu.LoadModule(km);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.InnerException.Message);
                throw;
            }

            Console.WriteLine("done");
        }

        [Cudafy]
        private static void Convolve(GThread thread, int pad, int stride, double[] biasesWeights,
            int filterWidth, int filterHeight, int filterDepth, double[] filterWeights,
            int inputWidth, int inputHeight, int inputDepth, double[] inputWeigths,
            int outputWidth, int outputHeight, int outputDepth, double[] outputWeights)
        {
            var t = thread.blockIdx.x * BlockSize + thread.threadIdx.x;
            var total = outputWidth * outputHeight;
            var temp = t % total;
            var depth = t / total;
            var ax = temp % outputWidth;
            var ay = temp / outputWidth;

            if (depth < outputDepth && ax < outputWidth && ay < outputHeight)
            {
                var totalfilter = filterHeight * filterWidth * filterDepth;

                var x = -pad + ax * stride;
                var y = -pad + ay * stride;

                // convolve centered at this particular location
                var a = 0.0;
                for (var fy = 0; fy < filterHeight; fy++)
                {
                    var oy = y + fy; // coordinates in the original input array coordinates
                    for (var fx = 0; fx < filterWidth; fx++)
                    {
                        var ox = x + fx;
                        if (oy >= 0 && oy < inputHeight && ox >= 0 && ox < inputWidth)
                        {
                            for (var fd = 0; fd < filterDepth; fd++)
                            {
                                //a += filter.Get(fx, fy, fd) * input.Get(ox, oy, fd);

                                a += filterWeights[depth * totalfilter + ((filterWidth * fy) + fx) * filterDepth + fd] *
                                     inputWeigths[((inputWidth * oy) + ox) * inputDepth + fd];
                            }
                        }
                    }
                }

                a += biasesWeights[depth];
                var ix = ((outputWidth * ay) + ax) * outputDepth + depth;
                outputWeights[ix] = a;
            }
        }

        private void AllocateGpu()
        {
            this.InitGpu();
            this.deviceInputWeigths = this.gpu.Allocate(this.InputActivation.Weights);
            this.deviceOutputWeights = this.gpu.Allocate<double>(this.OutputWidth * this.OutputHeight * this.OutputDepth);
            this.deviceBiasesWeights = this.gpu.Allocate(this.Biases.Weights);
            this.deviceFilterWeights = this.gpu.Allocate<double>(this.FilterCount * this.Width * this.Height * this.InputDepth);

            this.gpuAllocated = true;
        }

        public override Volume Forward(Volume input, bool isTraining = false)
        {
#if PERFORMANCE
            var chrono = Stopwatch.StartNew();
            var chronoTotal = Stopwatch.StartNew();
            chrono.Restart();
#endif
            this.InputActivation = input;
            var outputActivation = new Volume(this.OutputWidth, this.OutputHeight, this.OutputDepth, 0.0);

            var xyStride = this.Stride;

            if (!this.gpuAllocated)
            {
                this.AllocateGpu();
            }

            var coreCount = this.OutputHeight * this.OutputWidth * this.OutputDepth;
            var gridSize = (int)(Math.Ceiling(coreCount / (double)BlockSize));

#if PERFORMANCE
            Console.WriteLine("Phase #1:{0}", chrono.Elapsed.TotalMilliseconds);
            chrono.Restart();
#endif
            this.gpu.CopyToDevice(this.Biases.Weights, this.deviceBiasesWeights);
            this.gpu.CopyToDevice(input.Weights, this.deviceInputWeigths);
            this.gpu.CopyToDevice(outputActivation.Weights, this.deviceOutputWeights);

            var count = this.Width * this.Height * this.InputDepth;
            for (var i = 0; i < this.FilterCount; i++)
            {
                this.gpu.CopyToDevice(this.Filters[i].Weights, 0, this.deviceFilterWeights, i * count, count);
            }

#if PERFORMANCE
            Console.WriteLine("Phase #2:{0}", chrono.Elapsed.TotalMilliseconds);
            chrono.Restart();
#endif
            this.gpu.Launch(new dim3(gridSize), new dim3(BlockSize), Convolve, this.Pad, xyStride, this.deviceBiasesWeights,
                this.Width, this.Height, this.InputDepth, this.deviceFilterWeights,
                input.Width, input.Height, input.Depth, this.deviceInputWeigths,
                this.OutputWidth, this.OutputHeight, this.OutputDepth, this.deviceOutputWeights);

#if PERFORMANCE
            Console.WriteLine("Phase #3:{0}", chrono.Elapsed.TotalMilliseconds);
            chrono.Restart();
#endif

            this.gpu.CopyFromDevice(this.deviceOutputWeights, outputActivation.Weights);

#if PERFORMANCE
            Console.WriteLine("Phase #4:{0}", chrono.Elapsed.TotalMilliseconds);
            chrono.Restart();
#endif
            this.OutputActivation = outputActivation;
            return this.OutputActivation;
        }

        public override void Backward()
        {
            var volume = this.InputActivation;
            volume.WeightGradients = new double[volume.Weights.Length]; // zero out gradient wrt bottom data, we're about to fill it

            var volumeWidth = volume.Width;
            var volumeHeight = volume.Height;
            var volumeDepth = volume.Depth;
            var xyStride = this.Stride;

#if PARALLEL
            var locker = new object();
            Parallel.For(0, this.OutputDepth, () => new Volume(volumeWidth, volumeHeight, volumeDepth, 0), (depth, state, temp) =>
#else
            var temp = volume;
            for (var depth = 0; depth < this.OutputDepth; depth++)
#endif
            {
                var filter = this.Filters[depth];
                var y = -this.Pad;
                for (var ay = 0; ay < this.OutputHeight; y += xyStride, ay++)
                {
                    // xyStride
                    var x = -this.Pad;
                    for (var ax = 0; ax < this.OutputWidth; x += xyStride, ax++)
                    {
                        // xyStride

                        // convolve centered at this particular location
                        var chainGradient = this.OutputActivation.GetGradient(ax, ay, depth);
                        // gradient from above, from chain rule
                        for (var fy = 0; fy < filter.Height; fy++)
                        {
                            var oy = y + fy; // coordinates in the original input array coordinates
                            for (var fx = 0; fx < filter.Width; fx++)
                            {
                                var ox = x + fx;
                                if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth)
                                {
                                    for (var fd = 0; fd < filter.Depth; fd++)
                                    {
                                        filter.AddGradient(fx, fy, fd, volume.Get(ox, oy, fd) * chainGradient);
                                        temp.AddGradient(ox, oy, fd, filter.Get(fx, fy, fd) * chainGradient);
                                    }
                                }
                            }
                        }

                        this.Biases.WeightGradients[depth] += chainGradient;
                    }
                }

#if !PARALLEL
            }
#else
                return temp;
            }
                ,
                result =>
                {
                    lock (locker)
                    {
                        volume.AddGradientFrom(result);
                    }
                });
#endif
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            // required
            this.OutputDepth = this.FilterCount;

            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            this.OutputWidth = (int)Math.Floor((this.InputWidth + this.Pad * 2 - this.Width) / (double)this.Stride + 1);
            this.OutputHeight = (int)Math.Floor((this.InputHeight + this.Pad * 2 - this.Height) / (double)this.Stride + 1);

            // initializations
            var bias = this.BiasPref;
            this.Filters = new List<Volume>();

            for (var i = 0; i < this.OutputDepth; i++)
            {
                this.Filters.Add(new Volume(this.Width, this.Height, this.InputDepth));
            }

            this.Biases = new Volume(1, 1, this.OutputDepth, bias);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();
            for (var i = 0; i < this.OutputDepth; i++)
            {
                response.Add(new ParametersAndGradients
                {
                    Parameters = this.Filters[i].Weights,
                    Gradients = this.Filters[i].WeightGradients,
                    L2DecayMul = this.L2DecayMul,
                    L1DecayMul = this.L1DecayMul
                });
            }

            response.Add(new ParametersAndGradients
            {
                Parameters = this.Biases.Weights,
                Gradients = this.Biases.WeightGradients,
                L1DecayMul = 0.0,
                L2DecayMul = 0.0
            });

            return response;
        }
    }
}