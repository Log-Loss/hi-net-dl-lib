package dl_lib.dense;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

import java.util.Map;

public class Dense {

    private Activation activation;
    private WeightInit weightInit;
    private int outputDim;

    private void buildConfig(Map<String, String> config) {

        this.outputDim = Integer.valueOf(config.get("outputDim"));

        // activation section
        switch (config.get("activation")) {
            case "relu":
                this.activation = Activation.RELU;
                break;
            case "leakyrelu":
                this.activation = Activation.LEAKYRELU;
                break;
            case "tanh":
                this.activation = Activation.TANH;
                break;
            case "sigmoid":
                this.activation = Activation.SIGMOID;
                break;
            default:
                this.activation = Activation.IDENTITY;
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

    public Dense(Map<String, String> config) {
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addDenseLayer(int layerNum, NeuralNetConfiguration.ListBuilder listBuilder) {

        return listBuilder.layer(layerNum, new DenseLayer.Builder()
                .nOut(this.outputDim)
                .activation(this.activation)
                .weightInit(this.weightInit)
                .build());
    }

}
