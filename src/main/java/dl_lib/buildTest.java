package dl_lib;

import dl_lib.cnn.Convolution;
import dl_lib.core.InputLayer;
import dl_lib.core.Pooling;
import dl_lib.core.outputLayer;
import dl_lib.dense.Dense;
import dl_lib.utils.Output;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.HashMap;
import java.util.Map;

public class buildTest {

    public static void main(String args[]) throws Exception {
        // this is the global variable that should be specified first
        int batchSize = 64;

        InputType inputType = InputType.convolutionalFlat(28,28,1);

        Map<String, String> convConfig = new HashMap<String, String>();
        convConfig.put("filters", "1");
        convConfig.put("kernelSize", "3");
        convConfig.put("strides", "2");
        convConfig.put("padding", "0");

        Map<String, String> poolConfig = new HashMap<String, String>();
        poolConfig.put("kernelSize", "3");
        poolConfig.put("strides", "2");
        poolConfig.put("poolingType", "max");

        Map<String, String> denseConfig = new HashMap<String, String>();
        denseConfig.put("activation", "relu");
        denseConfig.put("weightInit", "xavier");
        denseConfig.put("outputDim", "500");

        Map<String, String> outputConfig = new HashMap<String, String>();
        outputConfig.put("activation", "softmax");
        outputConfig.put("lossFunction", "neg");
        outputConfig.put("outputNum", "10");

        Map<String, String> inputConfig = new HashMap<String, String>();
        inputConfig.put("iteration", "1");
        inputConfig.put("lr", "0.01");
        inputConfig.put("l2", "0.0002");


        System.out.println("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        System.out.println("Build model....");
        // the function fo the project starts here
        InputLayer inputLayer = new InputLayer(inputConfig);
        NeuralNetConfiguration.ListBuilder listBuilder = inputLayer.buildInputLayer();
        Output output = inputLayer.getOutput();

        Convolution convolution = new Convolution(output, convConfig);
        listBuilder = convolution.addConvLayer(listBuilder);
        output = convolution.getOutput();

        Pooling pooling = new Pooling(output, poolConfig);
        listBuilder = pooling.addPoolingLayer(listBuilder);
        output = pooling.getOutput();

        Dense dense = new Dense(output, denseConfig);
        listBuilder = dense.addDenseLayer(listBuilder);
        output = dense.getOutput();

        outputLayer outputLayer = new outputLayer(output, outputConfig);
        listBuilder = outputLayer.addOutputLayer(listBuilder);

        MultiLayerConfiguration conf = listBuilder
                .setInputType(inputType)
                .backprop(true).pretrain(false).build();

        // this conf is the final output of this part
        // following is the test
        System.out.println(conf);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<1; i++ ) {
            model.fit(mnistTrain);

            System.out.println("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);
            System.out.println(eval.stats());
            mnistTest.reset();
        }

        model.predict(mnistTest.next().getFeatureMatrix());
        System.out.println("****************finished********************");
    }
}
