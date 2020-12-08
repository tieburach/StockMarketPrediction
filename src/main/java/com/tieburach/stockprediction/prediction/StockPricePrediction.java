package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.config.NeuralNetProperties;
import com.tieburach.stockprediction.config.RecurrentNetwork;
import com.tieburach.stockprediction.model.DataEntity;
import com.tieburach.stockprediction.util.ExcelUtils;
import com.tieburach.stockprediction.util.GraphicsUtil;
import javafx.util.Pair;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;


@Component
public class StockPricePrediction {
    private static final Logger LOGGER = LoggerFactory.getLogger(StockPricePrediction.class);
    public static final int TOTAL_OUTCOMES = 1;
    public static final int VECTOR_SIZE = 4;
    private final NeuralNetProperties properties;
    private final RecurrentNetwork recurrentNetwork;
    private final double[] minValuesInFeature = new double[VECTOR_SIZE];
    private final double[] maxValuesInFeature = new double[VECTOR_SIZE];
    private MultiLayerNetwork bestNetwork;
    private double trainCorrelation;
    private double validationCorrelation;
    private double testCorrelation;

    @Autowired
    public StockPricePrediction(NeuralNetProperties properties, RecurrentNetwork recurrentNetwork) {
        this.properties = properties;
        this.recurrentNetwork = recurrentNetwork;
    }

    /**
     * Method to initialize minimum and maximum arrays
     *
     * @param wigDataEntities list of all entities
     */
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
        }
    }

    /**
     * Method that uses EarlyStoppingTrainer to end learning process
     *
     * @param entities stock data list
     */
    public void predictWithEarlyStopping(List<DataEntity> entities) {
        initializeMinAndMax(entities);
        double splitRatio = 0.7;
        int index = (int) Math.round(entities.size() * splitRatio);
        double splitRatio2 = 0.85;
        int index2 = (int) Math.round(entities.size() * splitRatio2);

        List<DataEntity> learningEntities = entities.subList(0, index);
        List<DataEntity> validationEntities = entities.subList(index, index2);
        List<DataEntity> testEntities = entities.subList(index2, entities.size());
        List<Pair<INDArray, INDArray>> testData = generateTestData(testEntities, properties.getDaysAhead());

        MultiLayerConfiguration myNetworkConfiguration = recurrentNetwork.getMultiLayerConfiguration(VECTOR_SIZE, TOTAL_OUTCOMES);
        DataSetIterator trainDataIterator = initializeIterator(learningEntities, properties.getDaysAhead());
        DataSetIterator validationDataIterator = initializeIterator(validationEntities, properties.getDaysAhead());
        DataSetIterator testDataIterator = initializeIterator(testEntities, properties.getDaysAhead());

        EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfig = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(50))
                .scoreCalculator(new DataSetLossCalculator(validationDataIterator, true))
                .evaluateEveryNEpochs(1)
                .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlyStoppingConfig, myNetworkConfiguration, trainDataIterator);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        bestNetwork = result.getBestModel();


        RegressionEvaluation regressionEvaluation = new RegressionEvaluation(1);
        int currentBatch = 0;
        trainCorrelation = 0;
        while (trainDataIterator.hasNext()) {
            DataSet next = trainDataIterator.next();
            INDArray output = bestNetwork.output(next.getFeatures());
            regressionEvaluation.eval(next.getLabels(), output);
            trainCorrelation += regressionEvaluation.averagecorrelationR2();
            currentBatch++;
        }
        trainCorrelation = trainCorrelation / currentBatch;
        validationDataIterator.reset();

        currentBatch = 0;
        validationCorrelation = 0;
        while (validationDataIterator.hasNext()) {
            DataSet next = validationDataIterator.next();
            INDArray output = bestNetwork.output(next.getFeatures());
            regressionEvaluation.eval(next.getLabels(), output);
            validationCorrelation += regressionEvaluation.averagecorrelationR2();
            currentBatch++;

        }
        validationCorrelation = validationCorrelation / currentBatch;
        validationDataIterator.reset();

        currentBatch = 0;
        testCorrelation = 0;
        while (testDataIterator.hasNext()) {
            DataSet next = testDataIterator.next();
            INDArray output = bestNetwork.output(next.getFeatures());
            regressionEvaluation.eval(next.getLabels(), output);
            testCorrelation += regressionEvaluation.averagecorrelationR2();
            currentBatch++;
        }
        testCorrelation = testCorrelation / currentBatch;
        testDataIterator.reset();


        LOGGER.info("Train correlation: " + trainCorrelation);
        LOGGER.info("Validation correlation: " + validationCorrelation);
        LOGGER.info("Test correlation: " + testCorrelation);

        predict(bestNetwork, true, testData);
    }


    /**
     * Method that looks on best mape and fixed number of epochs
     *
     * @param entities stock data list
     */
    public void predictWithoutEarlyStopping(List<DataEntity> entities) {
        initializeMinAndMax(entities);
        double splitRatio = 0.7;
        int index = (int) Math.round(entities.size() * splitRatio);
        double splitRatio2 = 0.85;
        int index2 = (int) Math.round(entities.size() * splitRatio2);

        List<DataEntity> learningEntities = entities.subList(0, index);
        List<DataEntity> validationEntities = entities.subList(index, index2);
        List<DataEntity> testEntities = entities.subList(index2, entities.size());
        List<Pair<INDArray, INDArray>> testData = generateTestData(testEntities, properties.getDaysAhead());

        MultiLayerConfiguration myNetworkConfiguration = recurrentNetwork.getMultiLayerConfiguration(VECTOR_SIZE, TOTAL_OUTCOMES);
        DataSetIterator trainDataIterator = initializeIterator(learningEntities, properties.getDaysAhead());
        DataSetIterator validationDataIterator = initializeIterator(validationEntities, properties.getDaysAhead());
        DataSetIterator testDataIterator = initializeIterator(testEntities, properties.getDaysAhead());

        MultiLayerNetwork network = new MultiLayerNetwork(myNetworkConfiguration);

        double bestMape = Double.MAX_VALUE;
        for (int i = 0; i < properties.getEpochs(); i++) {
            while (trainDataIterator.hasNext()) {
                network.fit(trainDataIterator.next());
            }
            trainDataIterator.reset();
            network.rnnClearPreviousState();
            LOGGER.info("Currently training epoch: " + i);
            double newMape = predict(network, false, testData);

            if (newMape < bestMape) {
                LOGGER.info("New best mape is:" + newMape);
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

        LOGGER.info("Train correlation: " + trainCorrelation);
        LOGGER.info("Validation correlation: " + validationCorrelation);
        LOGGER.info("Test correlation: " + testCorrelation);

        predict(bestNetwork, true, testData);
    }


    private List<Pair<INDArray, INDArray>> generateTestData(List<DataEntity> stockDataList, int number) {
        int bptt = properties.getBptt();
        int window = bptt + number;
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{bptt, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + bptt; j++) {
                DataEntity stock = stockDataList.get(j);
                input.putScalar(new int[]{j - i, 0}, (stock.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
                input.putScalar(new int[]{j - i, 1}, (stock.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
                input.putScalar(new int[]{j - i, 2}, (stock.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
                input.putScalar(new int[]{j - i, 3}, (stock.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
            }
            DataEntity stock = stockDataList.get(i + window - 1);
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
        LOGGER.info("Mape ERROR IN %: " + mape);
        if (draw) {
            GraphicsUtil.draw(predicts, actuals);
            ExcelUtils.writeToExcel(predicts, actuals);
        }
        return mape;
    }


    public StockDataSetIterator initializeIterator(List<DataEntity> entities, int number) {
        return new StockDataSetIterator(entities, properties.getBatchSize(), properties.getBptt(), number, minValuesInFeature, maxValuesInFeature);
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
