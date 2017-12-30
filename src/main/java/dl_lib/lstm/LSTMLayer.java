package dl_lib.lstm;

import dl_lib.utils.Input;
import dl_lib.utils.Output;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class LSTMLayer {

    private Input input;
    private Output output;
    private int hiddenLayerWidth = 50;
    private int hiddenLayerCount = 2;
    private int sequenceLen;

    private void buildInput(Output lastOutput) {
        this.input = new Input( lastOutput.getLayerNum() + 1 );
    }

    private void buildConfig(Map<String, String> config) {
        this.output = new Output(this.input.getLayerNum());
        this.hiddenLayerWidth = Integer.valueOf(config.get("hiddenLayerWidth"));
        this.hiddenLayerCount = Integer.valueOf(config.get("hiddenLayerCount"));
        this.sequenceLen = Integer.valueOf(config.get("sequenceLen"));
    }

    public LSTMLayer(Output lastOutput, Map<String, String> config) {
        this.buildInput(lastOutput);
        this.buildConfig(config);
    }

    public MultiLayerConfiguration addLSTMLayerAndOutput(NeuralNetConfiguration.ListBuilder listBuilder) {
        for (int i = 0; i < this.hiddenLayerCount; i++) {
            GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? this.sequenceLen : this.hiddenLayerWidth);
            hiddenLayerBuilder.nOut(this.hiddenLayerWidth);
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);

        outputLayerBuilder
                .activation(Activation.SOFTMAX)
                .nIn(this.hiddenLayerWidth)
                .nOut(this.sequenceLen);
        listBuilder.layer(this.hiddenLayerCount, outputLayerBuilder.build())
                .pretrain(false)
                .backprop(true);
        return listBuilder.build();
    }

    public Output getOutput() {
        return output;
    }
}
