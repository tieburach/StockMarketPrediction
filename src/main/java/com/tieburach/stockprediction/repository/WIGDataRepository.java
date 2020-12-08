package com.tieburach.stockprediction.repository;

import com.tieburach.stockprediction.model.DataEntity;
import com.tieburach.stockprediction.model.WIGDataEntity;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

import java.time.LocalDate;
import java.util.List;

public interface WIGDataRepository extends CrudRepository<WIGDataEntity, LocalDate> {
    @Query("SELECT A FROM DataEntity A")
    List<DataEntity> getAllFeatures();
}
