package dl_lib.cnn;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

import java.util.Map;

public class Convolution {
    private Input input;
    private Output output;
    private int filters;
    private int kernelSize;
    private int strides;
    private int padding;

    private void buildInput(Output lastOutput) {
        this.input = new Input( lastOutput.getLayerNum() + 1 );
    }

    private void buildConfig(Map<String, String> config) {

        this.output = new Output(this.input.getLayerNum());
        this.filters = Integer.valueOf(config.get("filters"));
        this.kernelSize = Integer.valueOf(config.get("kernelSize"));
        this.strides = Integer.valueOf(config.get("strides"));
        this.padding = Integer.valueOf(config.get("padding"));

    }

    public Convolution(Output output, Map<String, String> config) {
        this.buildInput(output);
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addConvLayer(NeuralNetConfiguration.ListBuilder listBuilder) {

        return listBuilder.layer(this.input.getLayerNum(), new ConvolutionLayer.Builder(this.kernelSize, this.kernelSize)
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                .nIn(input.getnChannels())
                .stride(this.strides, this.strides)
                .nOut(this.filters)
                .padding(this.padding, this.padding)
                .activation(Activation.IDENTITY)
                .build());
    }

    public Output getOutput() {
        return output;
    }
}
