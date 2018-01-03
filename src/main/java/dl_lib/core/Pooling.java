package dl_lib.core;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

import java.util.Map;

public class Pooling {

    private int kernelSize;
    private int strides;
    private PoolingType poolingType;

    private void buildConfig(Map<String, String> config) {

        this.kernelSize = Integer.valueOf(config.get("kernelSize"));
        this.strides = Integer.valueOf(config.get("strides"));

        if (config.get("poolingType").equals("max")) {
            this.poolingType = PoolingType.MAX;
        } else if (config.get("poolingType").equals("average")) {
            this.poolingType = PoolingType.AVG;
        }

    }

    public Pooling(Map<String, String> config) {
        this.buildConfig(config);
    }


    public NeuralNetConfiguration.ListBuilder addPoolingLayer(int layerNum, NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(layerNum, new SubsamplingLayer.Builder(this.poolingType)
                .kernelSize(this.kernelSize, this.kernelSize)
                .stride(this.strides, this.strides)
                .build());
    }

}
