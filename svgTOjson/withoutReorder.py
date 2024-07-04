import xml.etree.ElementTree as ET
import argparse
import pattern.wrapper4offset as para_pattern
import json
import re
import numpy as np
from decimal import Decimal, getcontext
from customconfig import Properties
import os

getcontext().prec = 1000

# use this file to generate the svg without offset
# Add templete in this version, and set props to generate dataset_properties.json for using in datasim.py

# In this file, I set px_to_vertices_scaling_factor all to 1.
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

def svg_to_jsonVertices(vertices):
    # convert svg coordinates to json coordinates
    # I use scaling_for_drawing=1, so I don't need to scale the vertices
    panel_vertices=np.array(vertices.copy())
    #panel_vertices=np.array(panel_vertices)-offset2    
    panel_vertices[:, 1] *= -1
    return panel_vertices.tolist()

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

def svg_to_jsonEdges(vertices,edge_order,curves_info):
    # here the vertices are in the svg coordinates
    result=[]
    vertices_num=len(vertices)
    vertices.append(vertices[0]) # add the first point to the end of the list to form a closed curve
    start=vertices[0]
    start_index=0
    end=vertices[0]
    end_index=0
    curve_index=0 # the index of the curve in curves_info
    for e in edge_order:
        if e=='M':
            continue
        elif e=='L':
            end_index=start_index+1
            if end_index==vertices_num:
                end_index=0
            result.append({"endpoints":[start_index,end_index]})
        elif e=='Q':
            end_index=start_index+1
            end=vertices[end_index]
            a,b=solve_curvature(start,end,curves_info[curve_index][0],curves_info[curve_index][1]) # a,b are the control points of the curve
            if end_index==vertices_num:
                end_index=0
            result.append({"endpoints":[start_index,end_index],"curvature":[a,b]})
            curve_index+=1
        start=vertices[end_index]
        start_index=end_index

    return result

'''
def panel_to_jsonElement(panel,offset2,panel_name,template): 
    # panel contain the information of a panel in svg file
    return3={} # return3 is the json element of the panel
    vertices=[]
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

    
    return3['vertices']=svg_to_jsonVertices(vertices,offset2)
    return3['edges']=svg_to_jsonEdges(vertices,edge_order,curves_info)   
    return3['translation']=template.spec['pattern']['panels'][panel_name]['translation']
    return3['rotation']=template.spec['pattern']['panels'][panel_name]['rotation']

    return max(np.max(np.array(vertices)[:, 0]), text_max_x), np.max(np.array(vertices)[:, 1]), return3
'''

def panel_to_jsonElement(panel,panel_name,template): 
    # panel contain the information of a panel in svg file
    return3={} # return3 is the json element of the panel
    vertices=[]
    pattern = re.compile(r'([MLQ])\s+([-\d.e\s]+)')
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

    
    return3['vertices']=svg_to_jsonVertices(vertices)
    return3['edges']=svg_to_jsonEdges(vertices,edge_order,curves_info)   
    return3['translation']=template.spec['pattern']['panels'][panel_name]['translation']
    return3['rotation']=template.spec['pattern']['panels'][panel_name]['rotation']

    return return3

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonPath', '-s', help='json file path for the garment', type=str)    
    args = parser.parse_args()
    
    
    #using for generate non offset svg
    pattern = para_pattern.VisPattern(args.jsonPath) # create a new pattern with minimal information

    directory = os.path.dirname(args.jsonPath)   
    pattern.serialize(path=directory,to_subfolder=False)
    