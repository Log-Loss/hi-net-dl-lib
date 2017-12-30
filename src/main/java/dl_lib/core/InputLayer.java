package dl_lib.core;

import dl_lib.utils.Output;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.Map;

// todo the constructor and set input type, link to dataset
public class InputLayer {

    private Output output;
    private int iteration;
    private double lr;
    private double l2 = 0.0;

    public InputLayer(Map<String, String> config) {
        this.output = new Output(-1);
        this.iteration = Integer.valueOf(config.get("iteration"));
        this.lr = Double.valueOf(config.get("lr"));

        if (config.containsKey("l2")) {
            this.l2 = Double.valueOf(config.get("l2"));
        }
    }


    public NeuralNetConfiguration.ListBuilder buildInputLayer() {
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(this.iteration) // Training iterations as above
                .regularization(true).l2(this.l2)
                .learningRate(this.lr)

//                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
//                .learningRateSchedule(lrSchedule)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list();

        return listBuilder;
    }

    public Output getOutput() {
        return output;
    }
}
