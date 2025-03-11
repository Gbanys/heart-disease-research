# resource "azurerm_virtual_network" "aks_vnet" {
#   name                = "aks-vnet"
#   address_space       = ["10.0.0.0/8"]
#   location            = azurerm_resource_group.aks_rg.location
#   resource_group_name = azurerm_resource_group.aks_rg.name
# }

# resource "azurerm_subnet" "aks_subnet" {
#   name                 = "aks-subnet"
#   resource_group_name  = azurerm_resource_group.aks_rg.name
#   virtual_network_name = azurerm_virtual_network.aks_vnet.name
#   address_prefixes     = ["10.240.0.0/16"]
# }

# resource "azurerm_subnet_nat_gateway_association" "aks_nat_association" {
#   subnet_id      = azurerm_subnet.aks_subnet.id
#   nat_gateway_id = azurerm_nat_gateway.aks_nat_gateway.id
# }

# resource "azurerm_public_ip" "aks_nat_public_ip" {
#   name                = "aks-nat-public-ip"
#   location            = azurerm_resource_group.aks_rg.location
#   resource_group_name = azurerm_resource_group.aks_rg.name
#   allocation_method   = "Static"
#   sku                 = "Standard"
# }

# resource "azurerm_nat_gateway" "aks_nat_gateway" {
#   name                = "aks-nat-gateway"
#   location            = azurerm_resource_group.aks_rg.location
#   resource_group_name = azurerm_resource_group.aks_rg.name
#   sku_name            = "Standard"
# }

# resource "azurerm_nat_gateway_public_ip_association" "example" {
#   nat_gateway_id       = azurerm_nat_gateway.aks_nat_gateway.id
#   public_ip_address_id = azurerm_public_ip.aks_nat_public_ip.id
# }

