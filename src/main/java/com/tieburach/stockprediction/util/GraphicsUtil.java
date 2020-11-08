package com.tieburach.stockprediction.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class GraphicsUtil {
    public static void draw(List<Double> predictedValues, List<Double> actualValues) {
        List<Integer> index = new ArrayList<>();
        for (int i = 0; i < predictedValues.size(); i++) {
            index.add(i);
        }
        int min = minValue(predictedValues, actualValues);
        int max = maxValue(predictedValues, actualValues);
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, index, predictedValues, "Predicts");
        addSeries(dataSet, index, actualValues, "Actuals");
        ChartFactory.setChartTheme(StandardChartTheme.createLegacyTheme());
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Prediction",
                "Index",
                "WIG20 - CLOSE",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        XYPlot xyPlot = chart.getXYPlot();
        final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
        domainAxis.setRange(index.get(0), index.get(index.size() - 1) + 2);
        domainAxis.setTickUnit(new NumberTickUnit(Math.floor((double) index.size() / 10)));
        domainAxis.setVerticalTickLabels(true);
        final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
        rangeAxis.setRange(min, max);
        rangeAxis.setTickUnit(new NumberTickUnit(Math.floor(((double) max - (double) min) / 10)));
        final ChartPanel panel = new ChartPanel(chart);
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        f.pack();
        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, List<Integer> x, List<Double> y, final String label) {
        final XYSeries s = new XYSeries(label);
        for (int j = 0; j < x.size(); j++) {
            s.add(x.get(j), y.get(j));
        }
        dataSet.addSeries(s);
    }

    private static int minValue(List<Double> predicts, List<Double> actuals) {
        double min = Integer.MAX_VALUE;
        for (int i = 0; i < predicts.size(); i++) {
            if (min > predicts.get(i)) {
                min = predicts.get(i);
            }
            if (min > actuals.get(i)) {
                min = actuals.get(i);
            }
        }
        return (int) (min * 0.95);
    }

    private static int maxValue(List<Double> predicts, List<Double> actuals) {
        double max = Integer.MIN_VALUE;
        for (int i = 0; i < predicts.size(); i++) {
            if (max < predicts.get(i)) {
                max = predicts.get(i);
            }
            if (max < actuals.get(i)) {
                max = actuals.get(i);
            }
        }
        return (int) (max * 1.05);
    }

}