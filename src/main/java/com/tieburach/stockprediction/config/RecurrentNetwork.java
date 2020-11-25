package com.tieburach.stockprediction.config;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class RecurrentNetwork {
    private static final double learningRate = 0.01;
    private static final int seed = 1;
    private static final int lstmLayer1Size = 8;
    private static final double dropoutRatio = 0.5;
    private final NeuralNetProperties properties;

    @Autowired
    public RecurrentNetwork(NeuralNetProperties properties) {
        this.properties = properties;
    }

    public MultiLayerConfiguration getMultiLayerConfiguration(int nIn, int nOut) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(lstmLayer1Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(lstmLayer1Size)
                        .nOut(nOut)
                        .activation(Activation.TANH)
                        .lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(properties.getBptt())
                .tBPTTBackwardLength(properties.getBptt())
                .pretrain(false)
                .backprop(true)
                .build();
    }
}
