package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.config.NeuralNetProperties;
import com.tieburach.stockprediction.config.RecurrentNetwork;
import com.tieburach.stockprediction.model.DataEntity;
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
import org.nd4j.linalg.dataset.DataSet;
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
    private final double splitRatio = 0.7;
    private final double splitRatio2 = 0.9;
    private final int totalOutcomes = 1;
    private final double[] minValuesInFeature = new double[VECTOR_SIZE];
    private final double[] maxValuesInFeature = new double[VECTOR_SIZE];
    private List<Pair<INDArray, INDArray>> testData;


    private void initializeMinAndMax(List<DataEntity> wigDataEntities) {
        for (int i = 0; i < maxValuesInFeature.length; i++) {
            maxValuesInFeature[i] = Double.MIN_VALUE;
            minValuesInFeature[i] = Double.MAX_VALUE;
        }

        for (DataEntity entity : wigDataEntities) {
            setMinAndMax(entity.getOpen(), 0);
            setMinAndMax(entity.getClose(), 1);
            setMinAndMax(entity.getLow(), 2);
            setMinAndMax(entity.getHigh(), 3);
//            setMinAndMax(entity.getDax_open(), 4);
//            setMinAndMax(entity.getDax_close(), 5);
//            setMinAndMax(entity.getDax_low(), 6);
//            setMinAndMax(entity.getDax_high(), 7);
//            setMinAndMax(entity.getSpx_open(), 8);
//            setMinAndMax(entity.getSpx_close(), 9);
//            setMinAndMax(entity.getSpx_low(), 10);
//            setMinAndMax(entity.getSpx_high(), 11);
        }
    }

    @Autowired
    public StockPricePrediction(NeuralNetProperties properties, RecurrentNetwork recurrentNetwork) {
        this.properties = properties;
        this.recurrentNetwork = recurrentNetwork;
    }

    public void initialize(List<DataEntity> entities) {
        initializeMinAndMax(entities);
        int index = (int) Math.round(entities.size() * splitRatio);
        int index2 = (int) Math.round(entities.size() * splitRatio2);


        List<DataEntity> learningEntities = entities.subList(0, index);
        List<DataEntity> validationEntities = entities.subList(index, index2);
        List<DataEntity> testEntities = entities.subList(index2, entities.size());
        generateTestData(testEntities);


        MultiLayerConfiguration myNetworkConfiguration = recurrentNetwork.getMultiLayerConfiguration(VECTOR_SIZE, totalOutcomes);
        DataSetIterator trainData = initializeIterator(learningEntities);
        DataSetIterator validationData = initializeIterator(validationEntities);

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(10))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(4, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(validationData, true))
                .evaluateEveryNEpochs(1)
                .build();


        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, myNetworkConfiguration, trainData);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        net = result.getBestModel();
    }

    private void generateTestData(List<DataEntity> stockDataList) {
        int bptt = properties.getBptt();
        int window = bptt + properties.getDaysAhead();
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{bptt, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + bptt; j++) {
                DataEntity stock = stockDataList.get(j);
                input.putScalar(new int[]{j - i, 0}, (stock.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
                input.putScalar(new int[]{j - i, 1}, (stock.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
                input.putScalar(new int[]{j - i, 2}, (stock.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
                input.putScalar(new int[]{j - i, 3}, (stock.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
//                input.putScalar(new int[]{j - i, 4}, (stock.getDax_open() - minValuesInFeature[4]) / (maxValuesInFeature[4] - minValuesInFeature[4]));
//                input.putScalar(new int[]{j - i, 5}, (stock.getDax_close() - minValuesInFeature[5]) / (maxValuesInFeature[5] - minValuesInFeature[5]));
//                input.putScalar(new int[]{j - i, 6}, (stock.getDax_low() - minValuesInFeature[6]) / (maxValuesInFeature[6] - minValuesInFeature[6]));
//                input.putScalar(new int[]{j - i, 7}, (stock.getDax_high() - minValuesInFeature[7]) / (maxValuesInFeature[7] - minValuesInFeature[7]));
//                input.putScalar(new int[]{j - i, 8}, (stock.getSpx_open() - minValuesInFeature[8]) / (maxValuesInFeature[8] - minValuesInFeature[8]));
//                input.putScalar(new int[]{j - i, 9}, (stock.getSpx_close() - minValuesInFeature[9]) / (maxValuesInFeature[9] - minValuesInFeature[9]));
//                input.putScalar(new int[]{j - i, 10}, (stock.getSpx_low() - minValuesInFeature[10]) / (maxValuesInFeature[10] - minValuesInFeature[10]));
//                input.putScalar(new int[]{j - i, 11}, (stock.getSpx_high() - minValuesInFeature[11]) / (maxValuesInFeature[11] - minValuesInFeature[11]));
            }
            DataEntity stock = stockDataList.get(i + window - 1);
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
        double mape = 0;
        for (int i = 0; i < predicts.length; i++) {
            mape += Math.abs((actuals[i] - predicts[i]) / actuals[i]);
        }
        mape = mape / predicts.length;
        System.out.println("Mape ERROR IN %: " + (mape * 100));

        GraphicsUtil.draw(predicts, actuals);
    }


    public StockDataSetIterator initializeIterator(List<DataEntity> entities) {
        return new StockDataSetIterator(entities, properties.getBatchSize(), properties.getBptt(), properties.getDaysAhead(), minValuesInFeature, maxValuesInFeature);
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
