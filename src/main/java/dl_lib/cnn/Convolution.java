package dl_lib.cnn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.Activation;

import java.util.Map;

public class Convolution {

    private int filters;
    private int kernelSize;
    private int strides;
    private int padding;
    private Activation activation;


    private void buildConfig(Map<String, String> config) {

        this.filters = Integer.valueOf(config.get("filters"));
        this.kernelSize = Integer.valueOf(config.get("kernelSize"));
        this.strides = Integer.valueOf(config.get("strides"));
        this.padding = Integer.valueOf(config.get("padding"));

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

    }

    public Convolution(Map<String, String> config) {
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addConvLayer(int layerNum, NeuralNetConfiguration.ListBuilder listBuilder) {

        return listBuilder.layer(layerNum, new ConvolutionLayer.Builder(this.kernelSize, this.kernelSize)
                .stride(this.strides, this.strides)
                .nOut(this.filters)
                .padding(this.padding, this.padding)
                .activation(this.activation)
                .build());
    }

}
