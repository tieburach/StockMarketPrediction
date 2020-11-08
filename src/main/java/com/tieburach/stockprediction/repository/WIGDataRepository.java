package com.tieburach.stockprediction.repository;

import com.tieburach.stockprediction.model.WIGDataEntity;
import org.springframework.data.repository.CrudRepository;

import java.time.LocalDate;
import java.util.List;

public interface WIGDataRepository extends CrudRepository<WIGDataEntity, LocalDate> {
    List<WIGDataEntity> getAllByDateBetween(LocalDate startDate, LocalDate endDate);
}
