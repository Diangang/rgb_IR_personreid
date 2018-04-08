--[[
   data.lua
   
   Copyright 2015 Arulkumar <arul.csecit@ymail.com>
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301, USA.
   
   
]]--

--[[-----------------------------------------------------------

preparation of data for training and testing 

data can be different types of datasets (ethz, cuhk03, viper)

---]]-------------------------------------------------------------

require 'lfs'
require 'utilities'
require 'io'
require 'torch'
require 'pl'

logger = require 'log'
logger.outfile = opt.logFile

--data buffer to be filled
trainData = {}
testData = {}
additionalGallery = {}

-- function to read the ids from txt file 
function read_ids_file(filepath)
    -- read the file contents
    local filehandle = io.open(filepath, 'r')
    local contents = filehandle:read()
    io.close(filehandle)
    
    print(contents)
    
    -- parse the contents into 4 digit ids
    local ids = contents:split(',')
    logger.info(filepath .. ' : ' .. #ids .. ' ids found')
    --io.read()
    
    local decoded_ids = {}
    for index, id in ipairs(ids) do
        table.insert(decoded_ids, string.format("%04d", id))
    end
    
    return decoded_ids
end

--to insert all the file paths to particular id config
function insert_into_id_config(current_table, cam_name, filenames, filepaths, cams)
    for index, filename in ipairs(filenames) do
        current_table[cam_name .. '-' .. filename] = read_image_and_resize(filepaths[index], cams[cam_name] == 'IR')
    end
    return current_table
end

function read_file_names_of_ids(ids, cams, datapath)
    local id_filepaths = {}
    local total_files = 0
    
    for index, id in ipairs(ids) do
        logger.trace("ID: " .. id)
        for cam_name, cam_category in pairs(cams) do
            current_cam_id_path = paths.concat(datapath, cam_name, id)
            if isFolderExists(current_cam_id_path) then
                -- if the id folder exists, read all file paths
                filenames, filepaths = getAllFileNamesInDir(current_cam_id_path)
                
                if id_filepaths[id] == nil then id_filepaths[id] = {}; end
                if id_filepaths[id]['rgb'] == nil then id_filepaths[id]['rgb'] = {}; end
                if id_filepaths[id]['IR'] == nil then id_filepaths[id]['IR'] = {}; end
                
                logger.info(current_cam_id_path .. " : " .. #filenames .. " files")
                total_files = total_files + #filenames
                id_filepaths[id][cam_category] = insert_into_id_config(id_filepaths[id][cam_category], cam_name, filenames, filepaths, cams)
            end
        end
        
        assert(table.map_length(id_filepaths[id]['rgb']) ~= 0, "no rgb files found for id : " .. id)
        assert(table.map_length(id_filepaths[id]['IR']) ~= 0, "no IR files found for id : " .. id)
    end
    
    return id_filepaths, total_files
end

if(opt.dataset == 'sysu_mm01') then
	--load cuhk03 dataset
	--get all file names
	
	cams = {['cam1'] = 'rgb',
            ['cam2'] = 'rgb',
            ['cam3'] = 'IR',
            ['cam4'] = 'rgb',
            ['cam5'] = 'rgb',
            ['cam6'] = 'IR',
            }
             
    experiment_configpath = opt.datapath..'exp'
    
    -- read experiment file settings (Train, validation, test)
    train_ids_file = paths.concat(experiment_configpath, 'train_id.txt')
    val_ids_file = paths.concat(experiment_configpath, 'val_id.txt')
    test_ids_file = paths.concat(experiment_configpath, 'test_id.txt')
    train_ids = read_ids_file(train_ids_file)
    val_ids = read_ids_file(val_ids_file)
    test_ids = read_ids_file(test_ids_file)
        
	train_id_files = {};
    val_id_files = {}
	test_id_files = {};
    
    configFile = paths.concat(opt.datapath,'data.config')
    
    if(path.exists(configFile) == false) then 
        logger.trace('previous configuration (train.config) for training NOT found in ' .. opt.save)
        
        --[[
        -- id ->
        --     rgb ->
                    ... filepaths
               IR ->
                    ... filepaths
        --]]
        train_id_files, total_train_files = read_file_names_of_ids(train_ids, cams, opt.datapath)
        val_id_files, total_val_files = read_file_names_of_ids(val_ids, cams, opt.datapath)
        test_id_files, total_test_files = read_file_names_of_ids(test_ids, cams, opt.datapath)
        logger.trace("total images files of rgb + IR (train/val/test) : " .. total_train_files .. '/' 
                      .. total_val_files .. '/' .. total_test_files)
        
        --save the configuration
        config = {}
        config['train'] = train_id_files;
        config['val'] = val_id_files;
        config['test'] = test_id_files;
        torch.save(configFile, config) -- 'ascii'
    else
        -- load the configuration
        logger.trace('previous configuration (train.config) for training found in ' .. opt.save)
        config = torch.load(configFile)
        train_id_files = config['train'];
        val_id_files = config['val'];
        test_id_files = config['test'];
    end
    --io.read()
    
	logger.trace('number of train, validation, test ids are ' .. table.map_length(train_id_files) .. 
                 ' / ' .. table.map_length(val_id_files) .. ' / ' .. table.map_length(test_id_files))

else

	logger.trace('unknown dataset!');

end
