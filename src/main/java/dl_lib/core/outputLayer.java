package dl_lib.core;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class outputLayer {

    private LossFunctions.LossFunction lossFunction;
    private Activation activation;
    private int outputNum;

    private void buildConfig(Map<String, String> config) {

        this.outputNum = Integer.valueOf(config.get("outputNum"));

        switch (config.get("activation")) {
            case "tanh":
                this.activation = Activation.TANH;
                break;
            case "softmax":
                this.activation = Activation.SOFTMAX;
                break;
            case "sigmoid":
                this.activation = Activation.SIGMOID;
                break;
            default:
                this.activation = Activation.IDENTITY;
        }

        if (config.get("lossFunction").equals("neg")) {
            this.lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
        } else if (config.get("lossFunction").equals("sqrt")) {
            this.lossFunction = LossFunctions.LossFunction.SQUARED_LOSS;
        }
    }

    public outputLayer(Map<String, String> config) {
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addOutputLayer(int layerNum, NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(layerNum, new OutputLayer.Builder(this.lossFunction)
                .nOut(outputNum)
                .activation(this.activation)
                .build());
    }
}
