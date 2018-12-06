### Rscript to pre-process TRaC data for ingestion by python

setwd("C:/Users/zool1232/Work/Code/LA_Example/")

trac.data <- read.csv("Data/Combined Haiti TRaC ALL PERSONS.csv")

valid.sample <- which(!is.na(trac.data$LSA1) & !is.na(trac.data$MSP1) & !is.na(trac.data$AMA1) & !is.na(trac.data$age) & !is.na(trac.data$NUM_SDE) & trac.data$age <= 50 & !(trac.data$SDE_lon > -72.47 & trac.data$SDE_lon < -72.25 & trac.data$SDE_lat > 18.425 & trac.data$SDE_lat < 18.65) & trac.data$age > 0)

Nclusters <- length(unique(trac.data$NUM_SDE[valid.sample]))

cluster.sdenums <- unique(trac.data$NUM_SDE[valid.sample])
cluster.longlats <- matrix(0,nrow=Nclusters,ncol=2)
for (i in 1:Nclusters) {cluster.longlats[i,] <- cbind(trac.data$SDE_lon,trac.data$SDE_lat)[valid.sample[trac.data$NUM_SDE[valid.sample]==cluster.sdenums[i]][1],]}

reference.image <- raster("CovariateRasters/covariates_accessibility_to_cities_2015_v1.0.tif")
valid.coordinates <- coordinates(reference.image)[!is.na(getValues(reference.image)),]
invalid.longlats <- which(is.na(extract(reference.image,cluster.longlats)))
for (i in invalid.longlats) {
  nearest.valid <- which.min((valid.coordinates[,1]-cluster.longlats[i,1])^2+(valid.coordinates[,2]-cluster.longlats[i,2])^2)
  cluster.longlats[i,] <- valid.coordinates[nearest.valid,]
}

cluster.membership <- match(trac.data$NUM_SDE[valid.sample],cluster.sdenums)

year <- c(2012,2015)[1+as.integer(trac.data$NUM_SDE[valid.sample]>62)]
age <- trac.data$age[valid.sample]
MSP <- trac.data$MSP1[valid.sample]
AMA <- trac.data$AMA1[valid.sample]
LSA <- trac.data$LSA1[valid.sample]

library(raster)
covariates <- matrix(NA,nrow=Nclusters,ncol=12)
covariates[,1] <- extract(raster("CovariateRasters/covariates_accessibility_to_cities_2015_v1.0.tif"),cluster.longlats)
covariates[,2] <- extract(raster("CovariateRasters/covariates_AI.tif"),cluster.longlats)
covariates[,3] <- extract(raster("CovariateRasters/covariates_DistToWater.tif"),cluster.longlats)
covariates[,4] <- extract(raster("CovariateRasters/covariates_Elevation.tif"),cluster.longlats)
covariates[,5] <- extract(raster("CovariateRasters/covariates_Landcover_forest.tif"),cluster.longlats)
covariates[,6] <- extract(raster("CovariateRasters/covariates_Landcover_grass_savanna.tif"),cluster.longlats)
covariates[,7] <- extract(raster("CovariateRasters/covariates_Landcover_urban_barren.tif"),cluster.longlats)
covariates[,8] <- extract(raster("CovariateRasters/covariates_Landcover_woodysavanna.tif"),cluster.longlats)
covariates[,9] <- extract(raster("CovariateRasters/covariates_OSM_v32.tif"),cluster.longlats)
covariates[,10] <- extract(raster("CovariateRasters/covariates_PET.tif"),cluster.longlats)
covariates[,11] <- extract(raster("CovariateRasters/covariates_Slope.tif"),cluster.longlats)
covariates[,12] <- extract(raster("CovariateRasters/covariates_TWI.tif"),cluster.longlats)

summary.data <- cbind(cluster.membership,cluster.longlats[cluster.membership,],year,age,MSP,AMA,LSA,covariates[cluster.membership,])
summary.data <- summary.data[sort.list(cluster.membership),]
colnames(summary.data) <- c("Cluster_Num","Longitude","Latitude","Year","Age","MSP","AMA","LSA","covariate_accessibility","covariate_AI","covariate_distTowater","covariate_elevation","covariate_forest","covariate_grass","covariate_urbanbarren","covariate_woodysavanna","covariate_OSM","covariate_PET","covariate_slope","covariate_TWI")
write.csv(summary.data,file="Data/summary_TRaC_data_with_covariates.csv")

library(INLA)
mesh <- inla.mesh.2d(cluster.longlats,max.edge=c(0.5,0.5),cut=0.01)
spde <- (inla.spde2.matern(mesh, alpha=2)$param.inla)[c("M0","M1","M2")]
A_matrix <- inla.mesh.project(mesh,cluster.longlats)$A
Afull_matrix <- inla.mesh.project(mesh,valid.coordinates)$A

library(Matrix)
writeMM(spde$M0,file="Data/M0_matrix.mtx")
writeMM(spde$M1,file="Data/M1_matrix.mtx")
writeMM(spde$M2,file="Data/M2_matrix.mtx")
writeMM(A_matrix,file="Data/A_matrix.mtx")
writeMM(Afull_matrix,file="Data/Afull_matrix.mtx")

covariates <- matrix(NA,nrow=length(valid.coordinates[,1]),ncol=12)
covariates[,1] <- extract(raster("CovariateRasters/covariates_accessibility_to_cities_2015_v1.0.tif"),valid.coordinates)
covariates[,2] <- extract(raster("CovariateRasters/covariates_AI.tif"),valid.coordinates)
covariates[,3] <- extract(raster("CovariateRasters/covariates_DistToWater.tif"),valid.coordinates)
covariates[,4] <- extract(raster("CovariateRasters/covariates_Elevation.tif"),valid.coordinates)
covariates[,5] <- extract(raster("CovariateRasters/covariates_Landcover_forest.tif"),valid.coordinates)
covariates[,6] <- extract(raster("CovariateRasters/covariates_Landcover_grass_savanna.tif"),valid.coordinates)
covariates[,7] <- extract(raster("CovariateRasters/covariates_Landcover_urban_barren.tif"),valid.coordinates)
covariates[,8] <- extract(raster("CovariateRasters/covariates_Landcover_woodysavanna.tif"),valid.coordinates)
covariates[,9] <- extract(raster("CovariateRasters/covariates_OSM_v32.tif"),valid.coordinates)
covariates[,10] <- extract(raster("CovariateRasters/covariates_PET.tif"),valid.coordinates)
covariates[,11] <- extract(raster("CovariateRasters/covariates_Slope.tif"),valid.coordinates)
covariates[,12] <- extract(raster("CovariateRasters/covariates_TWI.tif"),valid.coordinates)
write.csv(covariates,file="Data/fullcovariates.csv")

