package dl_lib.core;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

import java.util.Map;

public class Pooling {

    private Input input;
    private Output output;

    private int kernelSize;
    private int strides;
    private PoolingType poolingType;

    private void buildInput(Output lastOutput) {
        this.input = new Input( lastOutput.getLayerNum() + 1 );
    }

    private void buildConfig(Map<String, String> config) {

        this.output = new Output(this.input.getLayerNum());
        this.kernelSize = Integer.valueOf(config.get("kernelSize"));
        this.strides = Integer.valueOf(config.get("strides"));

        if (config.get("poolingType").equals("max")) {
            this.poolingType = PoolingType.MAX;
        } else if (config.get("poolingType").equals("average")) {
            this.poolingType = PoolingType.AVG;
        }

    }

    public Pooling(Output output, Map<String, String> config) {
        this.buildInput(output);
        this.buildConfig(config);
    }


    public NeuralNetConfiguration.ListBuilder addPoolingLayer(NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(input.getLayerNum(), new SubsamplingLayer.Builder(this.poolingType)
                .kernelSize(this.kernelSize, this.kernelSize)
                .stride(this.strides, this.strides)
                .build());
    }

    public Output getOutput() {
        return output;
    }
}
