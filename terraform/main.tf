# We strongly recommend using the required_providers block to set the
# Azure Provider source and version being used
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=4.1.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "aks_rg" {
  name     = "heart-disease-research-group"
  location = "UK South"
}

resource "azurerm_container_registry" "aks_container_registry" {
  name                = "akscontainerregistry81"
  resource_group_name = azurerm_resource_group.aks_rg.name
  location            = azurerm_resource_group.aks_rg.location
  sku                 = "Basic"
}

resource "azurerm_kubernetes_cluster" "aks_cluster" {
  name                = "heart-disease-research-cluster"
  location            = azurerm_resource_group.aks_rg.location
  resource_group_name = azurerm_resource_group.aks_rg.name
  dns_prefix          = "hdrakscluster"

  default_node_pool {
    name                 = "default"
    auto_scaling_enabled = true
    max_count            = 4
    min_count            = 1
    vm_size              = "Standard_DS2_v2"

    upgrade_settings {
      drain_timeout_in_minutes      = 0
      max_surge                     = "10%"
      node_soak_duration_in_minutes = 0
    }
  }

  identity {
    type = "SystemAssigned"
  }

  key_vault_secrets_provider{
    secret_rotation_enabled = false
  }
}

resource "azurerm_role_assignment" "aks_container_registry_role_assignment" {
  principal_id                     = azurerm_kubernetes_cluster.aks_cluster.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.aks_container_registry.id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "mlflow_role_assignment" {
  scope                = azurerm_storage_account.ml_storage_account.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_kubernetes_cluster.aks_cluster.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "aks_kv_role_assignment" {
  principal_id      = azurerm_kubernetes_cluster.aks_cluster.identity[0].principal_id
  role_definition_name = "Key Vault Secrets User"
  scope             = "/subscriptions/8d61b7e0-a087-4fd4-870a-83db896d37cd/resourceGroups/key-vault-group/providers/Microsoft.KeyVault/vaults/heart-disease-app-vault"
}


output "kube_config" {
  value     = azurerm_kubernetes_cluster.aks_cluster.kube_config_raw
  sensitive = true
}
