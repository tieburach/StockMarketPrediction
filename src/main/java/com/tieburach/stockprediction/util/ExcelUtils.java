package com.tieburach.stockprediction.util;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileOutputStream;
import java.time.LocalDateTime;

public class ExcelUtils {
    public static void writeToExcel(double[] predicts, double[] actuals) {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Prediction");
        Row row = sheet.createRow(0);
        Cell cell = row.createCell(0);
        cell.setCellValue("DATE");
        cell = row.createCell(1);
        cell.setCellValue("ACTUAL CLOSE");
        cell = row.createCell(2);
        cell.setCellValue("PREDICTED CLOSE");
        int rowCount = 1;
        for (int i = 0; i < predicts.length; i++) {
            row = sheet.createRow(rowCount);
            cell = row.createCell(0);
            cell.setCellValue("DATE");
            cell = row.createCell(1);
            cell.setCellValue(actuals[i]);
            cell = row.createCell(2);
            cell.setCellValue(predicts[i]);
            rowCount++;
        }
        for (int i = 0; i < 3; i++) {
            sheet.autoSizeColumn(i);
        }

        try (FileOutputStream outputStream = new FileOutputStream(
                "PREDICTION_" + LocalDateTime.now().getHour() + "_" + LocalDateTime.now().getMinute() + ".xlsx")) {
            workbook.write(outputStream);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
