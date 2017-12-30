package dl_lib.dense;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

import java.util.Map;

public class Dense {

    private Input input;
    private Output output;
    private Activation activation;
    private WeightInit weightInit;
    private int outputDim;

    private void buildInput(Output lastOutput) {
        this.input = new Input( lastOutput.getLayerNum() + 1);
    }

    private void buildConfig(Map<String, String> config) {

        this.output = new Output(this.input.getLayerNum());
        this.outputDim = Integer.valueOf(config.get("outputDim"));

        // activation section
        if (config.get("activation").equals("relu")) {
            this.activation = Activation.RELU;
        } else if (config.get("activation").equals("leakyrelu")) {
            this.activation = Activation.LEAKYRELU;
        } else if (config.get("activation").equals("tanh")) {
            this.activation = Activation.TANH;
        } else if (config.get("activation").equals("sigmoid")) {
            this.activation = Activation.SIGMOID;
        }

        // weight initialization section

        if (config.get("weightInit").equals("xavier")) {
            this.weightInit = WeightInit.XAVIER;
        } else if (config.get("weightInit").equals("zero")) {
            this.weightInit = WeightInit.ZERO;
        } else if (config.get("weightInitu").equals("uniform")) {
            this.weightInit = WeightInit.UNIFORM;
        }
    }

    public Dense(Output lastOutput, Map<String, String> config) {
        this.buildInput(lastOutput);
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addDenseLayer(NeuralNetConfiguration.ListBuilder listBuilder) {

        return listBuilder.layer(this.input.getLayerNum(), new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nOut(this.outputDim)
                .activation(this.activation)
                .weightInit(this.weightInit)
                .build());
    }

    // todo
    public Output getOutput() {
        return output;
    }
}
