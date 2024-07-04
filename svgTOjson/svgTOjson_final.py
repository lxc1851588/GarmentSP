import xml.etree.ElementTree as ET
import argparse
import pattern.wrappers as para_pattern
import json
import re
import math
import numpy as np
from decimal import Decimal, getcontext
from customconfig import Properties
import os

getcontext().prec = 1000
# panels_offset will be like {'Lback':[35,-30],'Rback':[-35,-30],'Rfront':[25,-30],'Lfront':[-25,-30]}, which records information of fixed points for all panels in a certain garment type
panels_offset={}

# Add templete in this version, and set props to generate dataset_properties.json for using in datasim.py
# Adding the inv scaling factor
# New: the converted json file has the same order of vertex and edge as the original json file

def read_svg(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    
    panel_names=[]
    panel_details=[]
    for path in root.findall('svg:path', namespaces):
        panel_details.append(path.attrib)
    for text in root.findall('svg:text', namespaces):
        panel_names.append(text.text)
    panels = dict(zip(panel_names, panel_details))  # please always keep the order of each panel. If use this please using python 3.7 or above
    
    return panels

def deal_scales(units_in_meter):
    if units_in_meter == 100:      # meters
        return 1/3
    else:
        return 1

# There are 3 cases for the offset_info:
def decode_offset_info(offset_info, vertices):
    offset_x=0
    offset_y=0
    if isinstance(offset_info[0], list):
        re1 = offset_info[0]
        re2 = offset_info[1]
        if math.isnan(re1[1]) and math.isnan(re2[1]): # Symmetric case
            fixed_index_y = re1[0]
            fixed_y=re1[2]
            offset_y=fixed_y-vertices[fixed_index_y][1]

            fixed_index_x = re1[0]
            reference_index_x = re2[0]
            sign=re2[2]
            if vertices[fixed_index_x][0]>vertices[reference_index_x][0] and sign==-1:
                dis=vertices[fixed_index_x][0]-vertices[reference_index_x][0]
                offset_x=dis/2.0-vertices[fixed_index_x][0]
            elif vertices[fixed_index_x][0]<vertices[reference_index_x][0] and sign==1:
                dis=vertices[reference_index_x][0]-vertices[fixed_index_x][0]
                offset_x=-dis/2.0-vertices[fixed_index_x][0]
            else:
                raise ValueError("The symmetric case has some problems")
        else:  # two fixed points case
            re_x=offset_info[0]
            re_y=offset_info[1]
            fixed_index_x = re_x[0]
            fixed_x=re_x[1]
            fixed_index_y = re_y[0]
            fixed_y=re_y[2]
            offset_x=fixed_x-vertices[fixed_index_x][0]
            offset_y=fixed_y-vertices[fixed_index_y][1]
    else:
        # The simplest case, only one fixed point, its xy coordinates indicate x,y offset 
        fixed_index_x = offset_info[0]
        fixed_index_y = offset_info[0]
        fixed_x=offset_info[1]
        fixed_y=offset_info[2]
        offset_x=fixed_x-vertices[fixed_index_x][0]
        offset_y=fixed_y-vertices[fixed_index_y][1]

    return offset_x, offset_y


def deal_panel_offset(panel_name,vertices):
    offset_info=panels_offset[panel_name]  # offset_info is a list of fixed points' information for that panel
    offset_x, offset_y=decode_offset_info(offset_info, vertices)    
    vertices=np.array(vertices)
    vertices[:,0]+=offset_x
    vertices[:,1]+=offset_y
    return vertices

def svg_to_jsonVertices(vertices,offset2,units_in_meter,template_edges):
    # convert svg coordinates to json coordinates
    inv_scaling_factor = deal_scales(units_in_meter)
    panel_vertices=vertices.copy()
    panel_vertices=np.array(panel_vertices)-offset2
    panel_vertices[:, 1] *= -1
    panel_vertices = panel_vertices * inv_scaling_factor

    panel_vertices_ordered = panel_vertices.copy()
    vertices_index=0
    for edge in template_edges:
        start=edge['endpoints'][0]
        panel_vertices_ordered[start][0]=panel_vertices[vertices_index][0]
        panel_vertices_ordered[start][1]=panel_vertices[vertices_index][1]        
        vertices_index+=1

    panel_vertices_final=deal_panel_offset(panel_name,panel_vertices_ordered)
    return panel_vertices_final.tolist()

def solve_curvature(start,end,control0,control1):
    start=np.array(start)
    end=np.array(end)
    edge=end-start
    edge_perp = np.array([-edge[1], edge[0]])
    control_point=np.array([control0,control1])

    # solve the matrix equation
    A = np.array([[edge[0], -edge_perp[0]], [edge[1], -edge_perp[1]]])
    b = control_point-start
    x = np.linalg.solve(A, b)

    return float(Decimal(x[0])), float(Decimal(x[1]))

def svg_to_jsonEdges(vertices,edge_order,curves_info,template_edges):
    # here the vertices are in the svg coordinates
    # vertices are panel ordered vertices
    # template_edges are the edges of this panel in the template
    result=[]
    curve_index=0 # the index of the curve in curves_info
    template_index=0 # the index of which edge is now in the template
    start_index=0
    end_index=0
    vertices.append(vertices[0])
    for e in edge_order:
        if e=='M':
            continue
        elif e=='L':
            end_index=start_index+1            
            result.append({"endpoints":[template_edges[template_index]['endpoints'][0],template_edges[template_index]['endpoints'][1]]})
        elif e=='Q':
            end_index=start_index+1            
            start=vertices[start_index]
            end=vertices[end_index]
            a,b=solve_curvature(start,end,curves_info[curve_index][0],curves_info[curve_index][1]) # a,b are the control points of the curve            
            result.append({"endpoints":[template_edges[template_index]['endpoints'][0],template_edges[template_index]['endpoints'][1]],"curvature":[a,b]})
            curve_index+=1

        start_index=end_index
        template_index+=1

    return result

def panel_to_jsonElement(panel,offset2,panel_name,template,template_panel): 
    # panel contain the information of a panel in svg file
    # template_panel is the corresponding template of this panel
    return3={} # return3 is the json element of the panel
    vertices=[] # every element in vertices is a list of two elements, which are the x and y coordinates of a vertex
    pattern = re.compile(r'([A-Za-z])\s+((?:\d+\.?\d*\s*)+)')
    matches = pattern.findall(panel)
    edge_order=[]
    curves_info=[]
    for match in matches:
        edge_order.append(match[0])
        if match[0]=='M':
            vertices.append([float(match[1].split()[0]),float(match[1].split()[1])])
        elif match[0]=='L':
            vertices.append([float(match[1].split()[0]),float(match[1].split()[1])])
        elif match[0]=='Q':
            vertices.append([float(match[1].split()[2]),float(match[1].split()[3])])
            curves_info.append([float(match[1].split()[0]),float(match[1].split()[1])])
    # discard the last point in vertices
    vertices.pop()

    # for the location of the panel offset
    panel_center = np.mean(vertices, axis=0)
    text_insert = panel_center + np.array([-25, 3])
    text_max_x = text_insert[0] + 10 * len(panel_name)
        
    return3['vertices']=svg_to_jsonVertices(vertices,offset2,template.spec['properties']['units_in_meter'],template_panel['edges'])
    return3['edges']=svg_to_jsonEdges(vertices,edge_order,curves_info,template_panel['edges'])   
    return3['translation']=template.spec['pattern']['panels'][panel_name]['translation']    
    return3['rotation']=template.spec['pattern']['panels'][panel_name]['rotation']

    return max(np.max(np.array(vertices)[:, 0]), text_max_x), np.max(np.array(vertices)[:, 1]), return3

# Replace None with np.nan when read in the panel_offset_dict json file
def replace_none_with_nan(data):
    if isinstance(data, dict):
        return {k: replace_none_with_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_none_with_nan(element) for element in data]
    elif data is None:
        return np.nan
    else:
        return data
    

if __name__ == "__main__":

    with open('./panel_offset_dict.json', 'r') as json_file:
        panel_offset_dict = json.load(json_file)

    # The dict to record the offsets for all garment panels
    panel_offset_dict = replace_none_with_nan(panel_offset_dict)

    parser = argparse.ArgumentParser()
    parser.add_argument('--svgPath', '-s', help='svg file path', type=str)
    args = parser.parse_args()
    
    garment=args.svgPath.split('/')[-1]
    garmentType=re.sub(r'_[A-Z0-9]+_pattern\.svg$', '', garment)
    if(garmentType not in panel_offset_dict.keys()):
        raise ValueError("The garment type is not in the panel_offset_dict.json file")
    panels_offset=panel_offset_dict[garmentType]
   
    templateDic=os.path.abspath(os.path.join(args.svgPath, "../../"))
    templatePath=os.path.join(templateDic, garmentType+'_template_specification.json')

    template = para_pattern.VisPattern(templatePath)
    pattern = para_pattern.VisPattern() # create a new pattern with minimal information


    panels=read_svg(args.svgPath)    
    pattern.spec['pattern']['panel_order']=list(panels.keys())
    

    base_offset = [60, 60]
    panel_offset_x = 0
    heights = [0]
    
    for panel_name, panel_information in panels.items():      
        panel_offset_x, height,pattern.spec['pattern']['panels'][panel_name]=panel_to_jsonElement(panel_information['d'],[panel_offset_x + base_offset[0], base_offset[1]],panel_name,template,template.spec['pattern']['panels'][panel_name])
        heights.append(height)
    
    pattern.spec['pattern']['stitches']=template.spec['pattern']['stitches']

    pattern.spec['parameters'] = template.spec['parameters']
    pattern.spec['parameter_order'] = template.spec['parameter_order']
    
    directory = os.path.dirname(args.svgPath)   
    json_output=json.dumps(pattern.spec, indent=4)
    with open(os.path.join(directory, 'specification.json'), 'w') as f:
        f.write(json_output)

    props = Properties()
    props.set_basic(
            templates=templatePath,
            #name='dress_1',
            size=1,   # number of patterns in the dataset based on the template
            to_subfolders=True)
    props.set_section_config('generator')
    props.add_sys_info()
    props.serialize(os.path.join(os.path.dirname(directory), 'dataset_properties.json'))

    # have a look at the json to svg
    # pattern = para_pattern.VisPattern(os.path.join(directory, 'specification.json'))
    # pattern.serialize(path=directory,to_subfolder=False)

