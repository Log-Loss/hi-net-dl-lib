package dl_lib.core;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

import java.util.Map;

public class Dropout {

    private Input input;
    private Output output;

    private double dropoutRate;

    private void buildInput(Output lastOutput) {
        this.input = new Input( lastOutput.getLayerNum() + 1 );
    }

    private void buildConfig(Map<String, String> config) {

        this.output = new Output(this.input.getLayerNum());
        this.dropoutRate = Integer.valueOf(config.get("dropoutRate"));
    }

    public Dropout(Output output, Map<String, String> config) {
        this.buildInput(output);
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addDropoutLayer(NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(this.input.getLayerNum(), new org.deeplearning4j.nn.conf.layers.DropoutLayer.Builder()
                .dropOut(this.dropoutRate)
                .build());
    }

    public Output getOutput() {
        return output;
    }
}
