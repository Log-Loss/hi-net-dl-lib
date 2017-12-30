package dl_lib.core;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class outputLayer {
    private Input input;
    private Output output;

    private LossFunctions.LossFunction lossFunction;
    private Activation activation;
    private int outputNum;

    private void buildInput(Output lastOutput) {
        this.input = new Input(lastOutput.getLayerNum() + 1);
    }

    private void buildConfig(Map<String, String> config) {

        this.output = new Output(this.input.getLayerNum());
        this.outputNum = Integer.valueOf(config.get("outputNum"));

        if (config.get("activation").equals("tanh")) {
            this.activation = Activation.TANH;
        } else if (config.get("activation").equals("softmax")) {
            this.activation = Activation.SOFTMAX;
        } else if (config.get("activation").equals("sigmoid")) {
            this.activation = Activation.SIGMOID;
        }

        if (config.get("lossFunction").equals("neg")) {
            this.lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
        } else if (config.get("lossFunction").equals("sqrt")) {
            this.lossFunction = LossFunctions.LossFunction.SQUARED_LOSS;
        }
    }

    public outputLayer(Output lastOutput, Map<String, String> config) {
        this.buildInput(lastOutput);
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addOutputLayer(NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(input.getLayerNum(), new OutputLayer.Builder(this.lossFunction)
                .nOut(outputNum)
                .activation(this.activation)
                .build());
    }
}
