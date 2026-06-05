-- Advanced JSON Modeler Plugin for Roblox Studio
-- Final version with full material support and BottomSurface

local plugin = plugin 
local HttpService = game:GetService("HttpService") 
local Selection = game:GetService("Selection") 
local ChangeHistoryService = game:GetService("ChangeHistoryService") 

-- Create the toolbar button
local toolbar = plugin:CreateToolbar("JSON Modeler") 
local openButton = toolbar:CreateButton("Open JSON Modeler", "Opens the JSON modeling interface", "rbxassetid://4458901886") 

-- Create the main widget
local widgetInfo = DockWidgetPluginGuiInfo.new(
	Enum.InitialDockState.Float,
	false,
	false,
	500,
	400,
	300,
	300
)
local widget = plugin:CreateDockWidgetPluginGui("JSONModeler", widgetInfo)
widget.Title = "JSON Modeler"
widget.Name = "JSONModelerWidget"

-- Main UI Frame
local mainFrame = Instance.new("Frame")
mainFrame.BackgroundColor3 = Color3.fromRGB(45, 45, 45)
mainFrame.Size = UDim2.new(1, 0, 1, 0)
mainFrame.Parent = widget

-- Tab Frame
local tabFrame = Instance.new("Frame")
tabFrame.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
tabFrame.Size = UDim2.new(1, -20, 1, -20)
tabFrame.Position = UDim2.new(0, 10, 0, 10)
tabFrame.Parent = mainFrame

-- Tab Buttons
local tabButtons = Instance.new("Frame")
tabButtons.BackgroundTransparency = 1
tabButtons.Size = UDim2.new(1, 0, 0, 30)
tabButtons.Parent = tabFrame

local editorTabButton = Instance.new("TextButton")
editorTabButton.Size = UDim2.new(0.5, 0, 1, 0)
editorTabButton.Position = UDim2.new(0, 0, 0, 0)
editorTabButton.Text = "Editor"
editorTabButton.TextColor3 = Color3.new(1, 1, 1)
editorTabButton.BackgroundColor3 = Color3.fromRGB(80, 80, 80)
editorTabButton.Parent = tabButtons

local historyTabButton = Instance.new("TextButton")
historyTabButton.Size = UDim2.new(0.5, 0, 1, 0)
historyTabButton.Position = UDim2.new(0.5, 0, 0, 0)
historyTabButton.Text = "History"
historyTabButton.TextColor3 = Color3.new(1, 1, 1)
historyTabButton.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
historyTabButton.Parent = tabButtons

-- Content Frame
local contentFrame = Instance.new("Frame")
contentFrame.BackgroundTransparency = 1
contentFrame.Size = UDim2.new(1, 0, 1, -30)
contentFrame.Position = UDim2.new(0, 0, 0, 30)
contentFrame.Parent = tabFrame

-- Editor Frame
local editorFrame = Instance.new("Frame")
editorFrame.Size = UDim2.new(1, 0, 1, 0)
editorFrame.Visible = true
editorFrame.Parent = contentFrame

local jsonScroll = Instance.new("ScrollingFrame")
jsonScroll.Size = UDim2.new(1, 0, 0.8, 0)
jsonScroll.BackgroundTransparency = 1
jsonScroll.Parent = editorFrame

local jsonEditor = Instance.new("TextBox")
jsonEditor.Size = UDim2.new(1, -10, 1, -10)
jsonEditor.Position = UDim2.new(0, 5, 0, 5)
jsonEditor.MultiLine = true
jsonEditor.TextWrapped = true
jsonEditor.Font = Enum.Font.Code
jsonEditor.TextSize = 14
jsonEditor.TextXAlignment = Enum.TextXAlignment.Left
jsonEditor.TextYAlignment = Enum.TextYAlignment.Top
jsonEditor.ClearTextOnFocus = false
jsonEditor.BackgroundColor3 = Color3.fromRGB(30, 30, 30)
jsonEditor.TextColor3 = Color3.new(1, 1, 1)
jsonEditor.Parent = jsonScroll

local buttonFrame = Instance.new("Frame")
buttonFrame.Size = UDim2.new(1, 0, 0.2, 0)
buttonFrame.Position = UDim2.new(0, 0, 0.8, 0)
buttonFrame.BackgroundTransparency = 1
buttonFrame.Parent = editorFrame

local loadButton = Instance.new("TextButton")
loadButton.Size = UDim2.new(0.3, -5, 0.9, 0)
loadButton.Position = UDim2.new(0, 0, 0, 0)
loadButton.Text = "Load"
loadButton.TextColor3 = Color3.new(1, 1, 1)
loadButton.BackgroundColor3 = Color3.fromRGB(70, 120, 70)
loadButton.Parent = buttonFrame

local saveButton = Instance.new("TextButton")
saveButton.Size = UDim2.new(0.3, -5, 0.9, 0)
saveButton.Position = UDim2.new(0.35, 0, 0, 0)
saveButton.Text = "Save"
saveButton.TextColor3 = Color3.new(1, 1, 1)
saveButton.BackgroundColor3 = Color3.fromRGB(70, 70, 120)
saveButton.Parent = buttonFrame

local generateButton = Instance.new("TextButton")
generateButton.Size = UDim2.new(0.3, -5, 0.9, 0)
generateButton.Position = UDim2.new(0.7, 0, 0, 0)
generateButton.Text = "Generate"
generateButton.TextColor3 = Color3.new(1, 1, 1)
generateButton.BackgroundColor3 = Color3.fromRGB(120, 70, 70)
generateButton.Parent = buttonFrame

-- History Frame
local historyFrame = Instance.new("Frame")
historyFrame.Size = UDim2.new(1, 0, 1, 0)
historyFrame.Visible = false
historyFrame.Parent = contentFrame

local historyScroll = Instance.new("ScrollingFrame")
historyScroll.Size = UDim2.new(1, 0, 1, -40)
historyScroll.BackgroundTransparency = 1
historyScroll.Parent = historyFrame

local historyButtons = Instance.new("UIListLayout")
historyButtons.Padding = UDim.new(0, 5)
historyButtons.Parent = historyScroll

local clearHistoryButton = Instance.new("TextButton")
clearHistoryButton.Size = UDim2.new(1, -10, 0, 30)
clearHistoryButton.Position = UDim2.new(0, 5, 1, -35)
clearHistoryButton.Text = "Clear History"
clearHistoryButton.TextColor3 = Color3.new(1, 1, 1)
clearHistoryButton.BackgroundColor3 = Color3.fromRGB(120, 70, 70)
clearHistoryButton.Parent = historyFrame

-- History storage
local HISTORY = {}

-- Utility Functions
local function getDecalFace(decal)
	local face = decal.Face
	if face == Enum.NormalId.Top then return "Top"
	elseif face == Enum.NormalId.Bottom then return "Bottom"
	elseif face == Enum.NormalId.Front then return "Front"
	elseif face == Enum.NormalId.Back then return "Back"
	elseif face == Enum.NormalId.Left then return "Left"
	elseif face == Enum.NormalId.Right then return "Right"
	else return "Unknown" end
end

local function getSurfaceType(surface)
	if surface == Enum.SurfaceType.Smooth then return "Smooth"
	elseif surface == Enum.SurfaceType.Glue then return "Glue"
	elseif surface == Enum.SurfaceType.Weld then return "Weld"
	elseif surface == Enum.SurfaceType.Studs then return "Studs"
	elseif surface == Enum.SurfaceType.Inlet then return "Inlet"
	elseif surface == Enum.SurfaceType.Universal then return "Universal"
	elseif surface == Enum.SurfaceType.Hinge then return "Hinge"
	elseif surface == Enum.SurfaceType.Motor then return "Motor"
	elseif surface == Enum.SurfaceType.SteppingMotor then return "SteppingMotor"
	elseif surface == Enum.SurfaceType.SmoothNoOutlines then return "SmoothNoOutlines"
	else return "Unknown" end
end

local function getMeshInfo(meshPart)
	return {
		MeshId = meshPart.MeshId,
		TextureId = meshPart.TextureID,
		Material = tostring(meshPart.Material),
		Transparency = meshPart.Transparency
	}
end

local function getSpecialMeshInfo(specialMesh)
	return {
		Scale = {specialMesh.Scale.X, specialMesh.Scale.Y, specialMesh.Scale.Z},
		MeshType = tostring(specialMesh.MeshType),
		MeshId = specialMesh.MeshId,
		TextureId = specialMesh.TextureId,
		Offset = {specialMesh.Offset.X, specialMesh.Offset.Y, specialMesh.Offset.Z}
	}
end

local function getPartMaterialInfo(part)
	local matInfo = {
		Material = tostring(part.Material),
		MaterialVariant = tostring(part.MaterialVariant),
		Reflectance = part.Reflectance
	}

	-- Handle PBR material attributes safely
	local success, attributes = pcall(function()
		return part:GetAttribute("CoreMaterialAttributes")
	end)

	if success and attributes then
		matInfo.Smoothness = attributes.Smoothness
		matInfo.Metalness = attributes.Metalness
	end

	return matInfo
end

local function getWeldConstraintInfo(weld)
	return {
		Part0 = weld.Part0 and weld.Part0:GetFullName() or nil,
		Part1 = weld.Part1 and weld.Part1:GetFullName() or nil,
		Enabled = weld.Enabled
	}
end

local function getHumanoidInfo(humanoid)
	return {
		WalkSpeed = humanoid.WalkSpeed,
		Health = humanoid.Health,
		MaxHealth = humanoid.MaxHealth
	}
end

local function updateHistoryUI()
	-- Clear existing buttons
	for _, child in ipairs(historyScroll:GetChildren()) do
		if child:IsA("TextButton") then child:Destroy() end
	end

	-- Add history items
	for i, item in ipairs(HISTORY) do
		if i <= 10 then -- Limit to 10 most recent
			local historyButton = Instance.new("TextButton")
			historyButton.Size = UDim2.new(1, -10, 0, 40)
			historyButton.Text = item.name
			historyButton.TextColor3 = Color3.new(1, 1, 1)
			historyButton.BackgroundColor3 = Color3.fromRGB(70, 70, 70)
			historyButton.Parent = historyScroll

			historyButton.MouseButton1Click:Connect(function()
				jsonEditor.Text = HttpService:JSONEncode(item.data)
				editorTabButton.BackgroundColor3 = Color3.fromRGB(80, 80, 80)
				historyTabButton.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
				editorFrame.Visible = true
				historyFrame.Visible = false
			end)
		end
	end
end

-- Core Functions
local function jsonToModel(jsonData, parent)
	parent = parent or workspace

	local function processObject(objData, parentObj)
		local objType = objData["$type"] or "Part"
		local newObj = Instance.new(objType)

		for key, value in pairs(objData) do
			if key ~= "$type" and key ~= "Children" then
				if type(value) == "table" then
					if key == "Color" or key == "Color3" then
						newObj[key] = Color3.fromRGB(value[1], value[2], value[3])
					elseif key == "Size" or key == "Position" then
						newObj[key] = Vector3.new(unpack(value))
					elseif key == "Rotation" or key == "Orientation" then
						newObj.Orientation = Vector3.new(unpack(value))
					elseif key == "Scale" and objType == "SpecialMesh" then
						newObj.Scale = Vector3.new(unpack(value))
					elseif key == "Offset" and objType == "SpecialMesh" then
						newObj.Offset = Vector3.new(unpack(value))
					end
				else
					pcall(function()
						if key == "Face" and (objType == "Decal" or objType == "Texture") then
							if value == "Top" then newObj.Face = Enum.NormalId.Top
							elseif value == "Bottom" then newObj.Face = Enum.NormalId.Bottom
							elseif value == "Front" then newObj.Face = Enum.NormalId.Front
							elseif value == "Back" then newObj.Face = Enum.NormalId.Back
							elseif value == "Left" then newObj.Face = Enum.NormalId.Left
							elseif value == "Right" then newObj.Face = Enum.NormalId.Right end
						elseif key == "TopSurface" and objType == "Part" then
							if value == "Smooth" then newObj.TopSurface = Enum.SurfaceType.Smooth
							elseif value == "Glue" then newObj.TopSurface = Enum.SurfaceType.Glue
							elseif value == "Weld" then newObj.TopSurface = Enum.SurfaceType.Weld
							elseif value == "Studs" then newObj.TopSurface = Enum.SurfaceType.Studs
							elseif value == "Inlet" then newObj.TopSurface = Enum.SurfaceType.Inlet
							elseif value == "Universal" then newObj.TopSurface = Enum.SurfaceType.Universal
							elseif value == "Hinge" then newObj.TopSurface = Enum.SurfaceType.Hinge
							elseif value == "Motor" then newObj.TopSurface = Enum.SurfaceType.Motor
							elseif value == "SteppingMotor" then newObj.TopSurface = Enum.SurfaceType.SteppingMotor
							elseif value == "SmoothNoOutlines" then newObj.TopSurface = Enum.SurfaceType.SmoothNoOutlines end
						elseif key == "BottomSurface" and objType == "Part" then
							if value == "Smooth" then newObj.BottomSurface = Enum.SurfaceType.Smooth
							elseif value == "Glue" then newObj.BottomSurface = Enum.SurfaceType.Glue
							elseif value == "Weld" then newObj.BottomSurface = Enum.SurfaceType.Weld
							elseif value == "Studs" then newObj.BottomSurface = Enum.SurfaceType.Studs
							elseif value == "Inlet" then newObj.BottomSurface = Enum.SurfaceType.Inlet
							elseif value == "Universal" then newObj.BottomSurface = Enum.SurfaceType.Universal
							elseif value == "Hinge" then newObj.BottomSurface = Enum.SurfaceType.Hinge
							elseif value == "Motor" then newObj.BottomSurface = Enum.SurfaceType.Motor
							elseif value == "SteppingMotor" then newObj.BottomSurface = Enum.SurfaceType.SteppingMotor
							elseif value == "SmoothNoOutlines" then newObj.BottomSurface = Enum.SurfaceType.SmoothNoOutlines end
						elseif key == "Material" then
							local raw = tostring(value)
							local cleanName = raw:gsub("^Enum%.Material%.", ""):gsub("%s+", ""):lower()
							local normalizedMaterial

							for _, mat in ipairs(Enum.Material:GetEnumItems()) do
								if mat.Name:lower() == cleanName then
									normalizedMaterial = mat
									break
								end
							end

							if normalizedMaterial then
								newObj.Material = normalizedMaterial
							else
								warn("Unknown material: " .. raw .. " → fallback to Plastic")
								newObj.Material = Enum.Material.Plastic
							end
						elseif key == "MeshType" and objType == "SpecialMesh" then
							newObj.MeshType = Enum.MeshType[value]
						elseif key == "MaterialVariant" then
							newObj.MaterialVariant = value
						elseif key == "Reflectance" then
							newObj.Reflectance = value
						elseif key == "Smoothness" then
							pcall(function() newObj:SetAttribute("CoreMaterialAttributes", {Smoothness = value}) end)
						elseif key == "Metalness" then
							pcall(function() newObj:SetAttribute("CoreMaterialAttributes", {Metalness = value}) end)
						else
							newObj[key] = value
						end
					end)
				end
			end
		end

		newObj.Parent = parentObj

		if objData.Children then
			for _, childData in ipairs(objData.Children) do
				processObject(childData, newObj)
			end
		end

		return newObj
	end

	if jsonData.Model then
		ChangeHistoryService:SetWaypoint("BeforeImportJSONModel")

		local model = Instance.new("Model")
		model.Name = jsonData.Model.Name or "JSONModel"

		for _, childData in ipairs(jsonData.Model.Children) do
			processObject(childData, model)
		end

		model.Parent = parent
		ChangeHistoryService:SetWaypoint("AfterImportJSONModel")

		-- Add to history
		table.insert(HISTORY, 1, {
			name = model.Name,
			data = jsonData
		})
		updateHistoryUI()

		return model
	end
	return nil
end

local function modelToJson(model)
	local jsonData = {
		Model = {
			Name = model.Name,
			Children = {}
		}
	}

	local function processObject(obj)
		local objData = {
			["$type"] = obj.ClassName,
			Name = obj.Name
		}

		if obj:IsA("BasePart") then
			-- Basic part properties
			objData.Size = {obj.Size.X, obj.Size.Y, obj.Size.Z}
			objData.Position = {obj.Position.X, obj.Position.Y, obj.Position.Z}
			objData.Rotation = {obj.Orientation.X, obj.Orientation.Y, obj.Orientation.Z}
			objData.Color = {
				math.floor(obj.Color.R * 255),
				math.floor(obj.Color.G * 255),
				math.floor(obj.Color.B * 255)
			}
			objData.Anchored = obj.Anchored
			objData.Transparency = obj.Transparency
			objData.TopSurface = getSurfaceType(obj.TopSurface)
			objData.BottomSurface = getSurfaceType(obj.BottomSurface)

			-- Material properties
			local matInfo = getPartMaterialInfo(obj)
			objData.Material = matInfo.Material
			objData.MaterialVariant = matInfo.MaterialVariant
			objData.Reflectance = matInfo.Reflectance

			-- PBR properties (if they exist)
			if matInfo.Smoothness ~= nil then
				objData.Smoothness = matInfo.Smoothness
			end
			if matInfo.Metalness ~= nil then
				objData.Metalness = matInfo.Metalness
			end

			if obj:IsA("MeshPart") then
				local meshInfo = getMeshInfo(obj)
				objData.MeshId = meshInfo.MeshId
				objData.TextureId = meshInfo.TextureId
			end
		elseif obj:IsA("SpecialMesh") then
			local meshInfo = getSpecialMeshInfo(obj)
			objData.MeshId = meshInfo.MeshId
			objData.TextureId = meshInfo.TextureId
			objData.MeshType = meshInfo.MeshType
			objData.Scale = meshInfo.Scale
			objData.Offset = meshInfo.Offset
		elseif obj:IsA("Decal") or obj:IsA("Texture") then
			objData.Texture = obj.Texture
			objData.Transparency = obj.Transparency
			objData.Color3 = {
				math.floor(obj.Color3.R * 255),
				math.floor(obj.Color3.G * 255),
				math.floor(obj.Color3.B * 255)
			}
			objData.Face = getDecalFace(obj)
			objData.Shiny = obj.Shiny
			objData.Specular = obj.Specular
			objData.ZIndex = obj.ZIndex
		elseif obj:IsA("WeldConstraint") then
			local weldInfo = getWeldConstraintInfo(obj)
			objData.Part0 = weldInfo.Part0
			objData.Part1 = weldInfo.Part1
			objData.Enabled = weldInfo.Enabled
		elseif obj:IsA("Humanoid") then
			local humanoidInfo = getHumanoidInfo(obj)
			objData.WalkSpeed = humanoidInfo.WalkSpeed
			objData.Health = humanoidInfo.Health
			objData.MaxHealth = humanoidInfo.MaxHealth
		elseif obj:IsA("Script") or obj:IsA("LocalScript") then
			objData.Source = obj.Source
		end

		if #obj:GetChildren() > 0 then
			objData.Children = {}
			for _, child in ipairs(obj:GetChildren()) do
				table.insert(objData.Children, processObject(child))
			end
		end

		return objData
	end

	for _, child in ipairs(model:GetChildren()) do
		table.insert(jsonData.Model.Children, processObject(child))
	end

	return jsonData
end

-- UI Events
editorTabButton.MouseButton1Click:Connect(function()
	editorTabButton.BackgroundColor3 = Color3.fromRGB(80, 80, 80)
	historyTabButton.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
	editorFrame.Visible = true
	historyFrame.Visible = false
end)

historyTabButton.MouseButton1Click:Connect(function()
	editorTabButton.BackgroundColor3 = Color3.fromRGB(60, 60, 60)
	historyTabButton.BackgroundColor3 = Color3.fromRGB(80, 80, 80)
	editorFrame.Visible = false
	historyFrame.Visible = true
	updateHistoryUI()
end)

loadButton.MouseButton1Click:Connect(function()
	local success, jsonData = pcall(function()
		return HttpService:JSONDecode(jsonEditor.Text)
	end)

	if success then
		local model = jsonToModel(jsonData)
		if model then
			Selection:Set({model})
			print("Model successfully loaded with all features!")
		else
			warn("Failed to create model from JSON")
		end
	else
		warn("Invalid JSON: " .. tostring(jsonData))
	end
end)

saveButton.MouseButton1Click:Connect(function()
	local selection = Selection:Get()
	if #selection == 1 and selection[1]:IsA("Model") then
		local jsonData = modelToJson(selection[1])
		jsonEditor.Text = HttpService:JSONEncode(jsonData)
		print("Model saved with all features!")
	else
		warn("Please select exactly one Model")
	end
end)

generateButton.MouseButton1Click:Connect(function()
	local template = {
		Model = {
			Name = "NewModel",
			Children = {
				{
					["$type"] = "Part",
					Name = "BasePart",
					Size = {4, 1, 4},
					Position = {0, 0.5, 0},
					Rotation = {0, 45, 0},
					Color = {0, 170, 255},
					Material = "Plastic",
					MaterialVariant = "",
					Reflectance = 0,
					Smoothness = 0.5,
					Metalness = 0,
					Anchored = true,
					Transparency = 0,
					TopSurface = "Smooth",
					BottomSurface = "Smooth",
					Children = {
						{
							["$type"] = "Decal",
							Name = "ExampleDecal",
							Texture = "rbxassetid://123456789",
							Transparency = 0,
							Color3 = {255, 255, 255},
							Face = "Front",
							Shiny = 20,
							Specular = false
						},
						{
							["$type"] = "WeldConstraint",
							Part0 = "BasePart",
							Part1 = "AnotherPart",
							Enabled = true
						},
						{
							["$type"] = "SpecialMesh",
							MeshId = "rbxassetid://11111111",
							TextureId = "rbxassetid://22222222",
							MeshType = "Head",
							Scale = {1, 1, 1},
							Offset = {0, 0, 0}
						}
					}
				},
				{
					["$type"] = "MeshPart",
					Name = "MeshExample",
					MeshId = "rbxassetid://123456789",
					TextureId = "rbxassetid://987654321",
					Material = "Plastic",
					Transparency = 0,
					Size = {2, 2, 2},
					Position = {0, 2, 0}
				},
				{
					["$type"] = "Humanoid",
					WalkSpeed = 16,
					Health = 100,
					MaxHealth = 100
				}
			}
		}
	}
	jsonEditor.Text = HttpService:JSONEncode(template)
end)

clearHistoryButton.MouseButton1Click:Connect(function()
	HISTORY = {}
	updateHistoryUI()
end)

openButton.Click:Connect(function()
	widget.Enabled = not widget.Enabled
end)

print("Advanced JSON Modeler Plugin loaded successfully with BottomSurface support!")
