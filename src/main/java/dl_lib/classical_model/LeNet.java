package dl_lib.classical_model;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

public class LeNet {
    private int nChannels; // Number of input channels
    private int outputNum; // The number of possible outcomes
    private int batchSize; // Test batch size
    private int nEpochs; // Number of training epochs
    private int iterations; // Number of training iterations
    private int seed; //

    public LeNet() {
        nChannels = 1;
        outputNum = 10;
        batchSize = 64;
        nEpochs = 1;
        iterations = 1;
        seed = 123;
    }

    public LeNet(Map<String, Integer> config) {
        nChannels = config.get("nChannels") == null? 1 : config.get("nChannels");
        outputNum = config.get("outputNum") == null? 10 : config.get("outputNum");
        batchSize = config.get("batchSize") == null? 64 : config.get("batchSize");
        nEpochs = config.get("nEpochs") == null? 1 : config.get("nEpochs");
        iterations = config.get("iterations") == null? 1 : config.get("iterations");
    }

    private MultiLayerConfiguration buildModelConfig() {
        Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(.01)//.biasLearningRate(0.02)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                //TODO move the input type to somewhere else
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

                return conf;
    }

    public MultiLayerConfiguration getModelConfig() {
        return buildModelConfig();
    }
}
