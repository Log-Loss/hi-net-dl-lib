package dl_lib.core;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;

import java.util.Map;

public class Dropout {

    private double dropoutRate;

    private void buildConfig(Map<String, String> config) {

        this.dropoutRate = Integer.valueOf(config.get("dropoutRate"));
    }

    public Dropout(Map<String, String> config) {
        this.buildConfig(config);
    }

    public NeuralNetConfiguration.ListBuilder addDropoutLayer(int layerNum, NeuralNetConfiguration.ListBuilder listBuilder) {
        return listBuilder.layer(layerNum, new DropoutLayer.Builder()
                .dropOut(this.dropoutRate)
                .build());
    }

}
