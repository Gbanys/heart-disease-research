resource "azurerm_storage_account" "ml_storage_account" {
  name                     = "mlartifactstore504"
  resource_group_name      = azurerm_resource_group.aks_rg.name
  location                 = azurerm_resource_group.aks_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "mlflow_artifact_container" {
  name                  = "mlflow"
  storage_account_name  = azurerm_storage_account.ml_storage_account.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "data_container" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.ml_storage_account.name
  container_access_type = "private"
}