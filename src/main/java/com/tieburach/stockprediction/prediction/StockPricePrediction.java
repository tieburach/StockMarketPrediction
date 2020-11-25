package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.config.NeuralNetProperties;
import com.tieburach.stockprediction.config.RecurrentNetwork;
import com.tieburach.stockprediction.model.WIGDataEntity;
import com.tieburach.stockprediction.util.ExcelUtils;
import com.tieburach.stockprediction.util.GraphicsUtil;
import javafx.util.Pair;
import org.deeplearning4j.eval.RegressionEvaluation;
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

import static com.tieburach.stockprediction.prediction.StockDataSetIterator.VECTOR_SIZE;

@Component
public class StockPricePrediction {
    private final NeuralNetProperties properties;
    private final RecurrentNetwork recurrentNetwork;
    private final double[] minValuesInFeature = new double[VECTOR_SIZE];
    private final double[] maxValuesInFeature = new double[VECTOR_SIZE];
    private MultiLayerNetwork bestNetwork;
    private double trainCorrelation = 0;
    private double validationCorrelation = 0;
    private double testCorrelation = 0;

    @Autowired
    public StockPricePrediction(NeuralNetProperties properties, RecurrentNetwork recurrentNetwork) {
        this.properties = properties;
        this.recurrentNetwork = recurrentNetwork;
    }

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

    public void initialize(List<WIGDataEntity> entities) {
        initializeMinAndMax(entities);
        double splitRatio = 0.7;
        int index = (int) Math.round(entities.size() * splitRatio);
        double splitRatio2 = 0.85;
        int index2 = (int) Math.round(entities.size() * splitRatio2);

        List<WIGDataEntity> learningEntities = entities.subList(0, index);
        List<WIGDataEntity> validationEntities = entities.subList(index, index2);
        List<WIGDataEntity> testEntities = entities.subList(index2, entities.size());
        List<Pair<INDArray, INDArray>> testData = generateTestData(testEntities);

        int totalOutcomes = 1;
        MultiLayerConfiguration myNetworkConfiguration = recurrentNetwork.getMultiLayerConfiguration(VECTOR_SIZE, totalOutcomes);
        DataSetIterator trainDataIterator = initializeIterator(learningEntities);
        DataSetIterator validationDataIterator = initializeIterator(validationEntities);
        DataSetIterator testDataIterator = initializeIterator(testEntities);

        MultiLayerNetwork network = new MultiLayerNetwork(myNetworkConfiguration);

        double bestMape = Double.MAX_VALUE;
        for (int i = 0; i < properties.getEpochs(); i++) {
            while (trainDataIterator.hasNext()) {
                network.fit(trainDataIterator.next());
            }
            trainDataIterator.reset();
            network.rnnClearPreviousState();

            System.out.println("Currently training epoch: " + i);

            double newMape = predict(network, false, testData);


            if (newMape < bestMape) {
                System.out.println("New best mape is:" + newMape);
                bestMape = newMape;
                bestNetwork = network.clone();

                RegressionEvaluation regressionEvaluation = new RegressionEvaluation(1);
                int currentBatch = 0;
                trainCorrelation = 0;
                while (trainDataIterator.hasNext()) {
                    DataSet next = trainDataIterator.next();
                    INDArray output = network.output(next.getFeatures());
                    regressionEvaluation.eval(next.getLabels(), output);
                    trainCorrelation += regressionEvaluation.averagecorrelationR2();
                    currentBatch++;
                }
                trainCorrelation = trainCorrelation / currentBatch;
                validationDataIterator.reset();

                regressionEvaluation = new RegressionEvaluation(1);
                currentBatch = 0;
                validationCorrelation = 0;
                while (validationDataIterator.hasNext()) {
                    DataSet next = validationDataIterator.next();
                    INDArray output = network.output(next.getFeatures());
                    regressionEvaluation.eval(next.getLabels(), output);
                    validationCorrelation += regressionEvaluation.averagecorrelationR2();
                    currentBatch++;

                }
                validationCorrelation = validationCorrelation / currentBatch;
                validationDataIterator.reset();

                regressionEvaluation = new RegressionEvaluation(1);
                currentBatch = 0;
                testCorrelation = 0;
                while (testDataIterator.hasNext()) {
                    DataSet next = testDataIterator.next();
                    INDArray output = network.output(next.getFeatures());
                    regressionEvaluation.eval(next.getLabels(), output);
                    testCorrelation += regressionEvaluation.averagecorrelationR2();
                    currentBatch++;
                }
                testCorrelation = testCorrelation / currentBatch;
                testDataIterator.reset();
            }
        }
        System.out.println("Train correlation: " + trainCorrelation);
        System.out.println("Validation correlation: " + validationCorrelation);
        System.out.println("Test correlation: " + testCorrelation);

        predict(bestNetwork, true, testData);
    }


    private List<Pair<INDArray, INDArray>> generateTestData(List<WIGDataEntity> stockDataList) {
        int bptt = properties.getBptt();
        int window = bptt + properties.getDaysAhead();
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{bptt, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + bptt; j++) {
                WIGDataEntity stock = stockDataList.get(j);
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
            WIGDataEntity stock = stockDataList.get(i + window - 1);
            INDArray label = Nd4j.create(new int[]{1}, 'f');
            label.putScalar(new int[]{0}, stock.getClose());
            test.add(new Pair<>(input, label));
        }
        return test;
    }

    public double predict(MultiLayerNetwork net, boolean draw, List<Pair<INDArray, INDArray>> testData) {
        double max = maxValuesInFeature[1];
        double min = minValuesInFeature[1];
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        int bptt = properties.getBptt();
        for (int i = 0; i < testData.size(); i++) {
            double actualValue = testData.get(i).getValue().getDouble(0);
            double predictedValue = net.rnnTimeStep(testData.get(i).getKey()).getDouble(bptt - 1);
            predicts[i] = predictedValue * (max - min) + min;
            actuals[i] = actualValue;
        }
        double mape = 0;
        for (int i = 0; i < predicts.length; i++) {
            mape += Math.abs((actuals[i] - predicts[i]) / actuals[i]);
        }
        mape = (mape / predicts.length) * 100;
//        System.out.println("Mape ERROR IN %: " + mape);

        if (draw) {
            GraphicsUtil.draw(predicts, actuals);
            ExcelUtils.writeToExcel(predicts, actuals);
        }
        return mape;
    }


    public StockDataSetIterator initializeIterator(List<WIGDataEntity> entities) {
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
