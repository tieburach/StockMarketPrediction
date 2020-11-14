package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.config.NeuralNetProperties;
import com.tieburach.stockprediction.config.RecurrentNetwork;
import com.tieburach.stockprediction.model.WIGDataEntity;
import com.tieburach.stockprediction.util.ExcelUtils;
import com.tieburach.stockprediction.util.GraphicsUtil;
import javafx.util.Pair;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.tieburach.stockprediction.prediction.StockDataSetIterator.VECTOR_SIZE;

@Component
public class StockPricePrediction {
    private MultiLayerNetwork net;
    private final NeuralNetProperties properties;
    private final RecurrentNetwork recurrentNetwork;
    private final double splitRatio = 0.8;
    private final double splitRatio2 = 0.9;
    private final int totalOutcomes = 1;
    private final double[] minValuesInFeature = new double[VECTOR_SIZE];
    private final double[] maxValuesInFeature = new double[VECTOR_SIZE];
    private List<Pair<INDArray, INDArray>> testData;


    private void initializeMinAndMax(List<WIGDataEntity> wigDataEntities) {
        for (int i = 0; i < maxValuesInFeature.length; i++) {
            maxValuesInFeature[i] = Double.MIN_VALUE;
            minValuesInFeature[i] = Double.MAX_VALUE;
        }

        for (WIGDataEntity entity : wigDataEntities) {
            setMinAndMax(entity.getOpen(), 0);
            setMinAndMax(entity.getClose(), 1);
            setMinAndMax(entity.getLow(), 2);
            setMinAndMax(entity.getHigh(), 3);
        }
    }

    @Autowired
    public StockPricePrediction(NeuralNetProperties properties, RecurrentNetwork recurrentNetwork) {
        this.properties = properties;
        this.recurrentNetwork = recurrentNetwork;
    }

    public void initialize(List<WIGDataEntity> entities) {
        initializeMinAndMax(entities);
        int index = (int) Math.round(entities.size() * splitRatio);
        int index2 = (int) Math.round(entities.size() * splitRatio2);

        List<WIGDataEntity> learningEntities = entities.subList(0, index);
        List<WIGDataEntity> validationEntities = entities.subList(index, index2);
        List<WIGDataEntity> testEntities = entities.subList(index2, entities.size());
        generateTestData(testEntities);


        MultiLayerConfiguration myNetworkConfiguration = recurrentNetwork.getMultiLayerConfiguration(VECTOR_SIZE, totalOutcomes);
        DataSetIterator myTrainData = initializeTestIterator(learningEntities);
        DataSetIterator myTestData = initializeValidationIterator(validationEntities);


        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(8))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(2, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(myTestData, true))
                .evaluateEveryNEpochs(1)
                .build();


        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, myNetworkConfiguration, myTrainData);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        net = result.getBestModel();
    }

    private void generateTestData(List<WIGDataEntity> stockDataList) {
        int bptt = properties.getBptt();
        int window = bptt + 1;
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{bptt, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + bptt; j++) {
                WIGDataEntity stock = stockDataList.get(j);
                input.putScalar(new int[]{j - i, 0}, (stock.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
                input.putScalar(new int[]{j - i, 1}, (stock.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
                input.putScalar(new int[]{j - i, 2}, (stock.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
                input.putScalar(new int[]{j - i, 3}, (stock.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
            }
            WIGDataEntity stock = stockDataList.get(i + bptt);
            INDArray label = Nd4j.create(new int[]{1}, 'f');
            label.putScalar(new int[]{0}, stock.getClose());
            test.add(new Pair<>(input, label));
        }
        testData = test;
    }

    public void predict() {
        double max = maxValuesInFeature[1];
        double min = minValuesInFeature[1];
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        int bptt = properties.getBptt();
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(bptt - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }

        ExcelUtils.writeToExcel(predicts, actuals);

        GraphicsUtil.draw(predicts, actuals);
    }


    public StockDataSetIterator initializeTestIterator(List<WIGDataEntity> entities) {
        return new StockDataSetIterator(entities, properties.getBatchSize(), properties.getBptt(), minValuesInFeature, maxValuesInFeature);
    }

    public StockDataSetIterator initializeValidationIterator(List<WIGDataEntity> entities) {
        return new StockDataSetIterator(entities, properties.getBatchSize(), properties.getBptt(), minValuesInFeature, maxValuesInFeature);
    }


    private void setMinAndMax(Double value, int i) {
        if (value > maxValuesInFeature[i]) {
            maxValuesInFeature[i] = value;
        }
        if (value < minValuesInFeature[i]) {
            minValuesInFeature[i] = value;
        }
    }

}
