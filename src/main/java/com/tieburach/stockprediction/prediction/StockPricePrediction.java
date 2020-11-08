package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.config.NeuralNetProperties;
import com.tieburach.stockprediction.config.RecurrentNetwork;
import com.tieburach.stockprediction.model.WIGDataEntity;
import com.tieburach.stockprediction.util.ExcelUtils;
import com.tieburach.stockprediction.util.GraphicsUtil;
import javafx.util.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class StockPricePrediction {
    private final NeuralNetProperties properties;
    private final RecurrentNetwork recurrentNetwork;

    @Autowired
    public StockPricePrediction(NeuralNetProperties properties, RecurrentNetwork recurrentNetwork) {
        this.properties = properties;
        this.recurrentNetwork = recurrentNetwork;
    }

    public void initialize(List<WIGDataEntity> list) {
        StockDataSetIterator iterator = new StockDataSetIterator(list, properties.getBatchSize(), properties.getBptt(), properties.getDaysToShow());
        MultiLayerNetwork net = recurrentNetwork.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        for (int i = 0; i < properties.getEpochs(); i++) {
            while (iterator.hasNext()) {
                net.fit(iterator.next());
            }
            iterator.reset();
            net.rnnClearPreviousState();
        }
        predict(net, iterator);
    }

    private void predict(MultiLayerNetwork net, StockDataSetIterator iterator) {
        List<Pair<INDArray, INDArray>> testData = iterator.getTestDataSet();
        INDArray max = Nd4j.create(iterator.getMaxValuesInFeature());
        INDArray min = Nd4j.create(iterator.getMinValuesInFeature());
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] predictsNormalized = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        int daysAhead = properties.getDaysAhead() - 1;
        int bptt = properties.getBptt();
        for (int i = 0; i < testData.size(); i++) {
            INDArray currentDataArray = testData.get(i).getKey();
            if (i >= daysAhead) {
                currentDataArray = modifyCurrentDataArray(currentDataArray, i, daysAhead, predictsNormalized);
            }
            predictsNormalized[i] = net.rnnTimeStep(currentDataArray).getRow(bptt - 1);
            predicts[i] = net.rnnTimeStep(currentDataArray).getRow(bptt - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }

        ExcelUtils.writeToExcel(predicts, actuals, iterator.getDatesList());

        List<Double> predictedForClose = new ArrayList<>();
        List<Double> actualForClose = new ArrayList<>();
        for (int i = 0; i < predicts.length; i++) {
            predictedForClose.add(predicts[i].getDouble(1));
            actualForClose.add(actuals[i].getDouble(1));
        }
        GraphicsUtil.draw(predictedForClose, actualForClose);
    }

    private INDArray modifyCurrentDataArray(INDArray currentDataArray, int i, int days, INDArray[] predicts) {
        INDArray input = Nd4j.create(new int[]{properties.getBptt(), StockDataSetIterator.VECTOR_SIZE}, 'f');
        int daysCounter = days;
        for (int j = 0; j < properties.getBptt(); j++) {
            if (j >= properties.getBptt() - days) {
                for (int k = 0; k < StockDataSetIterator.VECTOR_SIZE; k++) {
                    INDArray predict = predicts[i - daysCounter];
                    input.putScalar(new int[]{j, k}, predict.getDouble(k));
                }
                daysCounter--;
            } else {
                for (int k = 0; k < StockDataSetIterator.VECTOR_SIZE; k++) {
                    input.putScalar(new int[]{j, k}, currentDataArray.getDouble(j, k));
                }
            }
        }
        return input;
    }
}
