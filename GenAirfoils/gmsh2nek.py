#!/usr/bin/env python3
"""
gmsh2nek.py - Convert Gmsh mesh files to Nek5000 re2 format
Converted from Fortran to Python
"""

import numpy as np
import struct
import sys
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Container for mesh data"""
    num_dim: int = 0
    num_elem: int = 0
    eftot: int = 0
    total_node: int = 0
    total_line: int = 0
    total_quad: int = 0
    total_hex: int = 0
    bc_number: int = 0
    
    # Arrays
    node_xyz: Optional[np.ndarray] = None
    node_line: Optional[np.ndarray] = None
    node_quad: Optional[np.ndarray] = None
    node_hex: Optional[np.ndarray] = None
    
    line_array: Optional[np.ndarray] = None
    quad_array: Optional[np.ndarray] = None
    hex_array: Optional[np.ndarray] = None
    
    quad_line_array: Optional[np.ndarray] = None
    hex_face_array: Optional[np.ndarray] = None
    
    bc_id: Optional[np.ndarray] = None
    bc_char: Optional[List[str]] = None
    
    xm1: Optional[np.ndarray] = None
    ym1: Optional[np.ndarray] = None
    zm1: Optional[np.ndarray] = None
    
    ccurve: Optional[np.ndarray] = None
    curve: Optional[np.ndarray] = None
    cbc: Optional[np.ndarray] = None
    bc: Optional[np.ndarray] = None
    
    r_or_l: Optional[np.ndarray] = None


class GmshReader:
    """Read Gmsh mesh files"""
    
    def __init__(self):
        self.mesh = MeshData()
        
    def read_file_header(self, filename: str) -> Tuple[float, int]:
        """Read Gmsh file header to determine version and format"""
        with open(filename, 'r') as f:
            header = f.readline().strip()
            version_line = f.readline().strip().split()
            version = float(version_line[0])
            file_type = int(version_line[1])  # 0=ASCII, 1=Binary
            
        if version < 2.0 or version >= 3.0:
            raise ValueError("Only Gmsh version 2 mesh format is accepted")
            
        return version, file_type
    
    def read_2d_ascii(self, filename: str, preread: bool = False) -> MeshData:
        """Read 2D mesh in ASCII format"""
        # Create temporary copy
        temp_filename = filename + "_1"
        shutil.copy(filename, temp_filename)
        
        try:
            with open(filename, 'r') as f:
                mesh = self._parse_ascii_mesh(f, dimension=2, preread=preread)
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        return mesh
    
    def read_2d_binary(self, filename: str, preread: bool = False) -> MeshData:
        """Read 2D mesh in binary format"""
        with open(filename, 'rb') as f:
            mesh = self._parse_binary_mesh(f, dimension=2, preread=preread)
        return mesh
    
    def read_3d_ascii(self, filename: str, preread: bool = False) -> MeshData:
        """Read 3D mesh in ASCII format"""
        # Create temporary copy
        temp_filename = filename + "_1"
        shutil.copy(filename, temp_filename)
        
        try:
            with open(filename, 'r') as f:
                mesh = self._parse_ascii_mesh(f, dimension=3, preread=preread)
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        return mesh
    
    def read_3d_binary(self, filename: str, preread: bool = False) -> MeshData:
        """Read 3D mesh in binary format"""
        with open(filename, 'rb') as f:
            mesh = self._parse_binary_mesh(f, dimension=3, preread=preread)
        return mesh
    
    def _parse_ascii_mesh(self, f, dimension: int, preread: bool) -> MeshData:
        """Parse ASCII format mesh file"""
        mesh = MeshData()
        mesh.num_dim = dimension
        
        # Find PhysicalNames section
        for line in f:
            if line.strip() == "$PhysicalNames":
                break
        
        # Read physical names
        bc_number = int(f.readline())
        bc_id = np.zeros((2, bc_number), dtype=int)
        bc_char = []
        
        ibc_a = 0
        for i in range(bc_number):
            parts = f.readline().split()
            dim = int(parts[0])
            tag = int(parts[1])
            name = parts[2].strip('"')
            
            if (dimension == 2 and dim == 1) or (dimension == 3 and dim == 2):
                bc_id[0, ibc_a] = tag
                bc_char.append(name)
                ibc_a += 1
        
        mesh.bc_number = ibc_a
        mesh.bc_id = bc_id[:, :ibc_a]
        mesh.bc_char = bc_char[:ibc_a]
        
        # Find Nodes section
        for line in f:
            if line.strip() == "$Nodes":
                break
        
        # Read nodes
        total_node = int(f.readline())
        mesh.total_node = total_node
        node_xyz = np.zeros((3, total_node))
        
        for i in range(total_node):
            parts = f.readline().split()
            node_id = int(parts[0]) - 1  # Convert to 0-based
            node_xyz[0, node_id] = float(parts[1])
            node_xyz[1, node_id] = float(parts[2])
            node_xyz[2, node_id] = float(parts[3])
        
        mesh.node_xyz = node_xyz
        
        # Skip to Elements section
        for line in f:
            if line.strip() == "$Elements":
                break
        
        # Read elements
        total_elem = int(f.readline())
        
        if dimension == 2:
            mesh.node_line = np.zeros((2, total_node), dtype=int)
            mesh.node_quad = np.zeros((4, total_node), dtype=int)
            mesh.line_array = np.zeros((5, total_elem), dtype=int)
            mesh.quad_array = np.zeros((11, total_elem), dtype=int)
        else:
            mesh.node_quad = np.zeros((4, total_node), dtype=int)
            mesh.node_hex = np.zeros((8, total_node), dtype=int)
            mesh.quad_array = np.zeros((11, total_elem), dtype=int)
            mesh.hex_array = np.zeros((29, total_elem), dtype=int)
        
        total_line = 0
        total_quad = 0
        total_hex = 0
        
        for _ in range(total_elem):
            parts = f.readline().split()
            elem_id = int(parts[0])
            elem_type = int(parts[1])
            num_tags = int(parts[2])
            
            if dimension == 2:
                if elem_type == 8:  # Line3
                    mesh.line_array[0, total_line] = int(parts[3])  # Physical tag
                    mesh.line_array[1, total_line] = int(parts[4])  # Geometrical tag
                    for i in range(3):
                        mesh.line_array[2+i, total_line] = int(parts[5+i]) - 1
                    
                    # Update node_line
                    for i in range(2):
                        node_id = mesh.line_array[2+i, total_line]
                        self._add_to_node_line(mesh, node_id, total_line)
                    
                    # Update BC count
                    for ibc in range(mesh.bc_number):
                        if mesh.line_array[0, total_line] == mesh.bc_id[0, ibc]:
                            mesh.bc_id[1, ibc] += 1
                    
                    total_line += 1
                    
                elif elem_type == 16:  # Quad8
                    mesh.quad_array[0, total_quad] = int(parts[3])
                    mesh.quad_array[1, total_quad] = int(parts[4])
                    for i in range(8):
                        mesh.quad_array[2+i, total_quad] = int(parts[5+i]) - 1
                    
                    for i in range(4):
                        node_id = mesh.quad_array[2+i, total_quad]
                        self._add_to_node_quad(mesh, node_id, total_quad)
                    
                    total_quad += 1
                    
                elif elem_type == 10:  # Quad9
                    mesh.quad_array[0, total_quad] = int(parts[3])
                    mesh.quad_array[1, total_quad] = int(parts[4])
                    for i in range(9):
                        mesh.quad_array[2+i, total_quad] = int(parts[5+i]) - 1
                    
                    for i in range(4):
                        node_id = mesh.quad_array[2+i, total_quad]
                        self._add_to_node_quad(mesh, node_id, total_quad)
                    
                    total_quad += 1
                    
            else:  # 3D
                if elem_type == 16:  # Quad8
                    mesh.quad_array[0, total_quad] = int(parts[3])
                    mesh.quad_array[1, total_quad] = int(parts[4])
                    for i in range(8):
                        mesh.quad_array[2+i, total_quad] = int(parts[5+i]) - 1
                    
                    for i in range(4):
                        node_id = mesh.quad_array[2+i, total_quad]
                        self._add_to_node_quad(mesh, node_id, total_quad)
                    
                    # Update BC count
                    for ibc in range(mesh.bc_number):
                        if mesh.quad_array[0, total_quad] == mesh.bc_id[0, ibc]:
                            mesh.bc_id[1, ibc] += 1
                    
                    total_quad += 1
                    
                elif elem_type == 10:  # Quad9
                    mesh.quad_array[0, total_quad] = int(parts[3])
                    mesh.quad_array[1, total_quad] = int(parts[4])
                    for i in range(9):
                        mesh.quad_array[2+i, total_quad] = int(parts[5+i]) - 1
                    
                    for i in range(4):
                        node_id = mesh.quad_array[2+i, total_quad]
                        self._add_to_node_quad(mesh, node_id, total_quad)
                    
                    # Update BC count
                    for ibc in range(mesh.bc_number):
                        if mesh.quad_array[0, total_quad] == mesh.bc_id[0, ibc]:
                            mesh.bc_id[1, ibc] += 1
                    
                    total_quad += 1
                    
                elif elem_type == 17:  # Hex20
                    mesh.hex_array[0, total_hex] = int(parts[3])
                    mesh.hex_array[1, total_hex] = int(parts[4])
                    for i in range(20):
                        mesh.hex_array[2+i, total_hex] = int(parts[5+i]) - 1
                    
                    for i in range(8):
                        node_id = mesh.hex_array[2+i, total_hex]
                        self._add_to_node_hex(mesh, node_id, total_hex)
                    
                    total_hex += 1
                    
                elif elem_type == 12:  # Hex27
                    mesh.hex_array[0, total_hex] = int(parts[3])
                    mesh.hex_array[1, total_hex] = int(parts[4])
                    for i in range(27):
                        mesh.hex_array[2+i, total_hex] = int(parts[5+i]) - 1
                    
                    for i in range(8):
                        node_id = mesh.hex_array[2+i, total_hex]
                        self._add_to_node_hex(mesh, node_id, total_hex)
                    
                    total_hex += 1
        
        mesh.total_line = total_line
        mesh.total_quad = total_quad
        mesh.total_hex = total_hex
        
        if dimension == 2:
            mesh.num_elem = total_quad
        else:
            mesh.num_elem = total_hex
        
        # Trim arrays
        if dimension == 2:
            mesh.line_array = mesh.line_array[:, :total_line]
            mesh.quad_array = mesh.quad_array[:, :total_quad]
        else:
            mesh.quad_array = mesh.quad_array[:, :total_quad]
            mesh.hex_array = mesh.hex_array[:, :total_hex]
        
        if preread:
            logger.info(f"Total node number is {mesh.total_node}")
            if dimension == 2:
                logger.info(f"Total line element number is {mesh.total_line}")
                logger.info(f"Total quad element number is {mesh.total_quad}")
            else:
                logger.info(f"Total quad element number is {mesh.total_quad}")
                logger.info(f"Total hex element number is {mesh.total_hex}")
        
        return mesh
    
    def _parse_binary_mesh(self, f, dimension: int, preread: bool) -> MeshData:
        """Parse binary format mesh file"""
        mesh = MeshData()
        mesh.num_dim = dimension
        
        # Read header lines
        self._read_binary_line(f)
        self._read_binary_line(f)
        
        # Check endianness
        bone = struct.unpack('i', f.read(4))[0]
        f.read(1)  # Skip newline
        
        swap_endian = (bone != 1)
        
        # Find PhysicalNames section
        while True:
            line = self._read_binary_line(f)
            if line.strip() == "$PhysicalNames":
                break
        
        # Read physical names
        bc_number = int(self._read_binary_line(f))
        bc_id = np.zeros((2, bc_number), dtype=int)
        bc_char = []
        
        ibc_a = 0
        for i in range(bc_number):
            line = self._read_binary_line(f)
            parts = line.split()
            dim = int(parts[0])
            tag = int(parts[1])
            name = parts[2].strip('"')
            
            if (dimension == 2 and dim == 1) or (dimension == 3 and dim == 2):
                bc_id[0, ibc_a] = tag
                bc_char.append(name)
                ibc_a += 1
        
        mesh.bc_number = ibc_a
        mesh.bc_id = bc_id[:, :ibc_a]
        mesh.bc_char = bc_char[:ibc_a]
        
        # Find Nodes section
        while True:
            line = self._read_binary_line(f)
            if line.strip() == "$Nodes":
                break
        
        # Read nodes
        total_node = int(self._read_binary_line(f))
        mesh.total_node = total_node
        node_xyz = np.zeros((3, total_node))
        
        for i in range(total_node):
            data = f.read(4 + 3*8)  # int + 3 doubles
            if swap_endian:
                node_id = struct.unpack('>i', data[:4])[0] - 1
                coords = struct.unpack('>ddd', data[4:])
            else:
                node_id = struct.unpack('<i', data[:4])[0] - 1
                coords = struct.unpack('<ddd', data[4:])
            
            node_xyz[:, node_id] = coords
        
        mesh.node_xyz = node_xyz
        f.read(1)  # Skip newline
        
        # Skip to Elements section
        self._read_binary_line(f)  # $EndNodes
        self._read_binary_line(f)  # $Elements
        
        # Read elements
        total_elem = int(self._read_binary_line(f))
        
        if dimension == 2:
            mesh.node_line = np.zeros((2, total_node), dtype=int)
            mesh.node_quad = np.zeros((4, total_node), dtype=int)
            mesh.line_array = np.zeros((5, total_elem), dtype=int)
            mesh.quad_array = np.zeros((11, total_elem), dtype=int)
        else:
            mesh.node_quad = np.zeros((4, total_node), dtype=int)
            mesh.node_hex = np.zeros((8, total_node), dtype=int)
            mesh.quad_array = np.zeros((11, total_elem), dtype=int)
            mesh.hex_array = np.zeros((29, total_elem), dtype=int)
        
        total_line = 0
        total_quad = 0
        total_hex = 0
        
        elem_count = 0
        while elem_count < total_elem:
            # Read element header
            header = f.read(12)
            if swap_endian:
                elem_type, num_follow, num_tags = struct.unpack('>iii', header)
            else:
                elem_type, num_follow, num_tags = struct.unpack('<iii', header)
            
            if dimension == 2:
                if elem_type == 8:  # Line3
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 3))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+3}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+3}i', data)
                        
                        mesh.line_array[0, total_line] = values[1]  # Physical tag
                        mesh.line_array[1, total_line] = values[2]  # Geometrical tag
                        for i in range(3):
                            mesh.line_array[2+i, total_line] = values[3+i] - 1
                        
                        # Update node_line
                        for i in range(2):
                            node_id = mesh.line_array[2+i, total_line]
                            self._add_to_node_line(mesh, node_id, total_line)
                        
                        # Update BC count
                        for ibc in range(mesh.bc_number):
                            if mesh.line_array[0, total_line] == mesh.bc_id[0, ibc]:
                                mesh.bc_id[1, ibc] += 1
                        
                        total_line += 1
                        elem_count += 1
                        
                elif elem_type == 16:  # Quad8
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 8))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+8}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+8}i', data)
                        
                        mesh.quad_array[0, total_quad] = values[1]
                        mesh.quad_array[1, total_quad] = values[2]
                        for i in range(8):
                            mesh.quad_array[2+i, total_quad] = values[3+i] - 1
                        
                        for i in range(4):
                            node_id = mesh.quad_array[2+i, total_quad]
                            self._add_to_node_quad(mesh, node_id, total_quad)
                        
                        total_quad += 1
                        elem_count += 1
                        
                elif elem_type == 10:  # Quad9
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 9))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+9}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+9}i', data)
                        
                        mesh.quad_array[0, total_quad] = values[1]
                        mesh.quad_array[1, total_quad] = values[2]
                        for i in range(9):
                            mesh.quad_array[2+i, total_quad] = values[3+i] - 1
                        
                        for i in range(4):
                            node_id = mesh.quad_array[2+i, total_quad]
                            self._add_to_node_quad(mesh, node_id, total_quad)
                        
                        total_quad += 1
                        elem_count += 1
                        
            else:  # 3D
                if elem_type == 16:  # Quad8
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 8))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+8}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+8}i', data)
                        
                        mesh.quad_array[0, total_quad] = values[1]
                        mesh.quad_array[1, total_quad] = values[2]
                        for i in range(8):
                            mesh.quad_array[2+i, total_quad] = values[3+i] - 1
                        
                        for i in range(4):
                            node_id = mesh.quad_array[2+i, total_quad]
                            self._add_to_node_quad(mesh, node_id, total_quad)
                        
                        # Update BC count
                        for ibc in range(mesh.bc_number):
                            if mesh.quad_array[0, total_quad] == mesh.bc_id[0, ibc]:
                                mesh.bc_id[1, ibc] += 1
                        
                        total_quad += 1
                        elem_count += 1
                        
                elif elem_type == 10:  # Quad9
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 9))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+9}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+9}i', data)
                        
                        mesh.quad_array[0, total_quad] = values[1]
                        mesh.quad_array[1, total_quad] = values[2]
                        for i in range(9):
                            mesh.quad_array[2+i, total_quad] = values[3+i] - 1
                        
                        for i in range(4):
                            node_id = mesh.quad_array[2+i, total_quad]
                            self._add_to_node_quad(mesh, node_id, total_quad)
                        
                        # Update BC count
                        for ibc in range(mesh.bc_number):
                            if mesh.quad_array[0, total_quad] == mesh.bc_id[0, ibc]:
                                mesh.bc_id[1, ibc] += 1
                        
                        total_quad += 1
                        elem_count += 1
                        
                elif elem_type == 17:  # Hex20
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 20))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+20}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+20}i', data)
                        
                        mesh.hex_array[0, total_hex] = values[1]
                        mesh.hex_array[1, total_hex] = values[2]
                        for i in range(20):
                            mesh.hex_array[2+i, total_hex] = values[3+i] - 1
                        
                        for i in range(8):
                            node_id = mesh.hex_array[2+i, total_hex]
                            self._add_to_node_hex(mesh, node_id, total_hex)
                        
                        total_hex += 1
                        elem_count += 1
                        
                elif elem_type == 12:  # Hex27
                    for _ in range(num_follow):
                        data = f.read(4 * (1 + num_tags + 27))
                        if swap_endian:
                            values = struct.unpack(f'>{1+num_tags+27}i', data)
                        else:
                            values = struct.unpack(f'<{1+num_tags+27}i', data)
                        
                        mesh.hex_array[0, total_hex] = values[1]
                        mesh.hex_array[1, total_hex] = values[2]
                        for i in range(27):
                            mesh.hex_array[2+i, total_hex] = values[3+i] - 1
                        
                        for i in range(8):
                            node_id = mesh.hex_array[2+i, total_hex]
                            self._add_to_node_hex(mesh, node_id, total_hex)
                        
                        total_hex += 1
                        elem_count += 1
        
        mesh.total_line = total_line
        mesh.total_quad = total_quad
        mesh.total_hex = total_hex
        
        if dimension == 2:
            mesh.num_elem = total_quad
        else:
            mesh.num_elem = total_hex
        
        # Trim arrays
        if dimension == 2:
            mesh.line_array = mesh.line_array[:, :total_line]
            mesh.quad_array = mesh.quad_array[:, :total_quad]
        else:
            mesh.quad_array = mesh.quad_array[:, :total_quad]
            mesh.hex_array = mesh.hex_array[:, :total_hex]
        
        if preread:
            logger.info(f"Total node number is {mesh.total_node}")
            if dimension == 2:
                logger.info(f"Total line element number is {mesh.total_line}")
                logger.info(f"Total quad element number is {mesh.total_quad}")
            else:
                logger.info(f"Total quad element number is {mesh.total_quad}")
                logger.info(f"Total hex element number is {mesh.total_hex}")
        
        return mesh
    
    def _read_binary_line(self, f) -> str:
        """Read a line from binary file"""
        line = b""
        while True:
            char = f.read(1)
            if char == b'\n' or not char:
                break
            line += char
        return line.decode('ascii')
    
    def _add_to_node_line(self, mesh: MeshData, node_id: int, line_id: int):
        """Add line to node's line list"""
        for i in range(2):
            if mesh.node_line[i, node_id] == 0:
                mesh.node_line[i, node_id] = line_id
                return
    
    def _add_to_node_quad(self, mesh: MeshData, node_id: int, quad_id: int):
        """Add quad to node's quad list"""
        for i in range(4):
            if mesh.node_quad[i, node_id] == 0:
                mesh.node_quad[i, node_id] = quad_id
                return
    
    def _add_to_node_hex(self, mesh: MeshData, node_id: int, hex_id: int):
        """Add hex to node's hex list"""
        for i in range(8):
            if mesh.node_hex[i, node_id] == 0:
                mesh.node_hex[i, node_id] = hex_id
                return


class MeshConverter:
    """Convert Gmsh mesh to Nek5000 format"""
    
    def __init__(self):
        self.mesh = None
        self.start_quad = 0
        self.start_hex = 0
        
        # Mapping arrays
        self.msh_to_nek_right = [0, 2, 8, 6, 1, 5, 7, 3, 4]  # 1-based to 0-based
        self.msh_to_nek_left = [2, 0, 6, 8, 1, 3, 7, 5, 4]
        self.msh_to_nek_3d = [0, 2, 8, 6, 18, 20, 26, 24, 1, 3, 9, 5, 11,
                              7, 17, 15, 19, 21, 23, 25, 4, 10, 12, 14, 16, 22, 13]
        
        self.quad_face_node_right = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.quad_face_node_left = [[1, 0], [0, 3], [3, 2], [2, 1]]
        
        self.hex_face_node = [
            [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4],
            [0, 1, 2, 3], [4, 5, 6, 7]
        ]
        
        # Edge definitions for 3D
        self.e3 = [
            [0, 1, 2], [2, 5, 8], [8, 7, 6], [6, 3, 0],
            [18, 19, 20], [20, 23, 26], [26, 25, 24], [24, 21, 18],
            [0, 9, 18], [2, 11, 20], [8, 17, 26], [6, 15, 24]
        ]
    
    def convert_2d(self, mesh: MeshData, start_quad: int = 0):
        """Convert 2D mesh elements"""
        self.mesh = mesh
        self.start_quad = start_quad
        
        # Detect right or left hand elements
        mesh.r_or_l = np.zeros(mesh.num_elem, dtype=int)
        
        for iquad in range(mesh.total_quad):
            # Detect handedness
            mesh.r_or_l[iquad + start_quad] = self._detect_r_or_l(iquad)
            
            # Map vertices
            if mesh.r_or_l[iquad + start_quad] == 0:  # Right hand
                for imsh in range(9):
                    inek = self.msh_to_nek_right[imsh]
                    mesh.xm1[inek, 0, 0, iquad + start_quad] = mesh.node_xyz[0, mesh.quad_array[imsh + 2, iquad]]
                    mesh.ym1[inek, 0, 0, iquad + start_quad] = mesh.node_xyz[1, mesh.quad_array[imsh + 2, iquad]]
            else:  # Left hand
                for imsh in range(9):
                    inek = self.msh_to_nek_left[imsh]
                    mesh.xm1[inek, 0, 0, iquad + start_quad] = mesh.node_xyz[0, mesh.quad_array[imsh + 2, iquad]]
                    mesh.ym1[inek, 0, 0, iquad + start_quad] = mesh.node_xyz[1, mesh.quad_array[imsh + 2, iquad]]
        
        # Search for boundaries
        for iquad in range(mesh.total_quad):
            for iline in range(4):
                # Get line nodes
                if mesh.r_or_l[iquad + start_quad] == 0:
                    lnode = [mesh.quad_array[self.quad_face_node_right[iline][j] + 2, iquad] for j in range(2)]
                else:
                    lnode = [mesh.quad_array[self.quad_face_node_left[iline][j] + 2, iquad] for j in range(2)]
                
                # Find matching line
                ifound = self._find_line(mesh, lnode)
                if ifound >= 0:
                    mesh.quad_line_array[iline, iquad + start_quad] = mesh.line_array[0, ifound]
        
        # Assign boundary conditions
        for iquad in range(mesh.total_quad):
            for iline in range(4):
                if mesh.quad_line_array[iline, iquad + start_quad] != 0:
                    mesh.cbc[iline, iquad + start_quad] = 'MSH'
                    mesh.bc[4, iline, iquad + start_quad] = mesh.quad_line_array[iline, iquad + start_quad]
    
    def convert_3d(self, mesh: MeshData, start_hex: int = 0):
        """Convert 3D mesh elements"""
        self.mesh = mesh
        self.start_hex = start_hex
        
        # Map vertices
        for ihex in range(mesh.total_hex):
            for imsh in range(27):
                inek = self.msh_to_nek_3d[imsh]
                mesh.xm1[inek, 0, 0, ihex + start_hex] = mesh.node_xyz[0, mesh.hex_array[imsh + 2, ihex]]
                mesh.ym1[inek, 0, 0, ihex + start_hex] = mesh.node_xyz[1, mesh.hex_array[imsh + 2, ihex]]
                mesh.zm1[inek, 0, 0, ihex + start_hex] = mesh.node_xyz[2, mesh.hex_array[imsh + 2, ihex]]
        
        # Search for boundaries using new O(N) scheme
        for iquad in range(mesh.total_quad):
            physical_tag = mesh.quad_array[0, iquad]
            quad_nodes = [mesh.quad_array[i + 2, iquad] for i in range(4)]
            
            # Find all hexes that share nodes with this quad
            hex_candidates = set()
            for node in quad_nodes:
                for i in range(8):
                    if mesh.node_hex[i, node] > 0:
                        hex_candidates.add(mesh.node_hex[i, node])
            
            # Check each candidate hex
            for ihex in hex_candidates:
                # Check all faces of this hex
                for iface in range(6):
                    face_nodes = [mesh.hex_array[self.hex_face_node[iface][j] + 2, ihex] for j in range(4)]
                    
                    # Check if face matches quad
                    if self._quad_match(face_nodes, quad_nodes):
                        mesh.hex_face_array[iface, ihex + start_hex] = physical_tag
                        break
        
        # Assign boundary conditions
        for ihex in range(mesh.total_hex):
            for iface in range(6):
                if mesh.hex_face_array[iface, ihex + start_hex] != 0:
                    mesh.cbc[iface, ihex + start_hex] = 'MSH'
                    mesh.bc[4, iface, ihex + start_hex] = mesh.hex_face_array[iface, ihex + start_hex]
    
    def _detect_r_or_l(self, quad_id: int) -> int:
        """Detect if quad is right-handed or left-handed"""
        nodes = [self.mesh.quad_array[i + 2, quad_id] for i in range(4)]
        
        # Vectors from node 0 to nodes 1 and 3
        vec12 = self.mesh.node_xyz[:2, nodes[1]] - self.mesh.node_xyz[:2, nodes[0]]
        vec14 = self.mesh.node_xyz[:2, nodes[3]] - self.mesh.node_xyz[:2, nodes[0]]
        
        # Cross product z-component
        cz = vec12[0] * vec14[1] - vec12[1] * vec14[0]
        
        return 0 if cz > 0 else 1  # 0 = right hand, 1 = left hand
    
    def _find_line(self, mesh: MeshData, lnode: List[int]) -> int:
        """Find line element with given nodes"""
        for iline in range(mesh.total_line):
            line_nodes = [mesh.line_array[i + 2, iline] for i in range(2)]
            if self._line_match(lnode, line_nodes):
                return iline
        return -1
    
    def _line_match(self, nodes1: List[int], nodes2: List[int]) -> bool:
        """Check if two lines have the same nodes"""
        return set(nodes1) == set(nodes2)
    
    def _quad_match(self, nodes1: List[int], nodes2: List[int]) -> bool:
        """Check if two quads have the same nodes"""
        return set(nodes1) == set(nodes2)
    
    def set_periodicity(self, mesh: MeshData, field_type: int = 1):
        """Set periodic boundary conditions interactively"""
        logger.info("******************************************************")
        logger.info("Boundary info summary")
        logger.info("BoundaryName     BoundaryID")
        for i in range(mesh.bc_number):
            logger.info(f"{mesh.bc_char[i]}    {mesh.bc_id[0, i]}")
        logger.info("******************************************************")
        
        nbc = int(input("Enter number of periodic boundary surface pairs: "))
        if nbc <= 0:
            return
        
        nface = 2 * mesh.num_dim
        
        for ibc in range(nbc):
            ptags = [int(x) for x in input("Input surface 1 and surface 2 BoundaryID: ").split()]
            pvec = [float(x) for x in input("Input translation vector (surface 1 -> surface 2): ").split()]
            pvec = np.array(pvec)
            
            # Find elements with matching tags
            pairs1 = []
            pairs2 = []
            
            if field_type == 1:
                elem_range = mesh.eftot
            else:
                elem_range = mesh.num_elem
            
            for iel in range(elem_range):
                for iface in range(nface):
                    if mesh.bc[4, iface, iel] == ptags[0]:
                        pairs1.append((iel, iface))
                    elif mesh.bc[4, iface, iel] == ptags[1]:
                        pairs2.append((iel, iface))
            
            if len(pairs1) != len(pairs2):
                logger.error(f"Mapping surface {ptags[0]} with {len(pairs1)} faces")
                logger.error(f"to surface {ptags[1]} with {len(pairs2)} faces")
                logger.error("ERROR: face numbers are not matching")
                continue
            
            # Match periodic faces
            ptol = 1e-5
            
            for iel1, iface1 in pairs1:
                # Get face center
                if mesh.num_dim == 2:
                    if mesh.r_or_l[iel1] == 0:
                        nodes = [mesh.quad_array[self.quad_face_node_right[iface1][j] + 2, iel1] for j in range(2)]
                    else:
                        nodes = [mesh.quad_array[self.quad_face_node_left[iface1][j] + 2, iel1] for j in range(2)]
                    center1 = np.mean([mesh.node_xyz[:, n] for n in nodes], axis=0)
                else:
                    nodes = [mesh.hex_array[self.hex_face_node[iface1][j] + 2, iel1] for j in range(4)]
                    center1 = np.mean([mesh.node_xyz[:, n] for n in nodes], axis=0)
                
                # Find matching face
                min_dist = float('inf')
                best_match = None
                
                for iel2, iface2 in pairs2:
                    # Get face center
                    if mesh.num_dim == 2:
                        if mesh.r_or_l[iel2] == 0:
                            nodes = [mesh.quad_array[self.quad_face_node_right[iface2][j] + 2, iel2] for j in range(2)]
                        else:
                            nodes = [mesh.quad_array[self.quad_face_node_left[iface2][j] + 2, iel2] for j in range(2)]
                        center2 = np.mean([mesh.node_xyz[:, n] for n in nodes], axis=0)
                    else:
                        nodes = [mesh.hex_array[self.hex_face_node[iface2][j] + 2, iel2] for j in range(4)]
                        center2 = np.mean([mesh.node_xyz[:, n] for n in nodes], axis=0)
                    
                    dist = np.linalg.norm(center2 - center1 - pvec)
                    
                    if dist < min_dist:
                        min_dist = dist
                        if dist <= ptol:
                            best_match = (iel2, iface2)
                
                if best_match:
                    iel2, iface2 = best_match
                    mesh.bc[0, iface1, iel1] = float(iel2)
                    mesh.bc[1, iface1, iel1] = float(iface2)
                    mesh.bc[0, iface2, iel2] = float(iel1)
                    mesh.bc[1, iface2, iel2] = float(iface1)
                    mesh.cbc[iface1, iel1] = 'P  '
                    mesh.cbc[iface2, iel2] = 'P  '
            
            # Check for errors
            nperror = 0
            for iel, iface in pairs1:
                if mesh.cbc[iface, iel] != 'P  ':
                    nperror += 1
            
            if nperror > 0:
                logger.error(f"Doing periodic check for surface {ptags[0]}")
                logger.error(f"ERROR: {nperror} faces did not map")
        
        logger.info("******************************************************")
        logger.info("Please set boundary conditions to all non-periodic boundaries")
        logger.info("in .usr file usrdat2() subroutine")
        logger.info("******************************************************")


class Re2Writer:
    """Write Nek5000 re2 format files"""
    
    def __init__(self, mesh: MeshData):
        self.mesh = mesh
        self.file = None
    
    def write(self, filename: str):
        """Write re2 file"""
        logger.info(f"\nWriting {filename}")
        
        with open(filename, 'wb') as self.file:
            self._write_header()
            self._write_xyz()
            self._write_curve()
            self._write_bc()
    
    def _write_header(self):
        """Write re2 header"""
        nbc_re2 = 1
        if self.mesh.num_elem != self.mesh.eftot:
            nbc_re2 = 2
        
        # Write header string
        header = f"#v004{self.mesh.num_elem:16d}{self.mesh.num_dim:3d}{self.mesh.eftot:16d}{nbc_re2:4d} hdr"
        header = header[:80].ljust(80)
        self.file.write(header.encode('ascii'))
        
        # Write endian discriminator
        test = np.float32(6.54321)
        self.file.write(test.tobytes())
    
    def _write_xyz(self):
        """Write element coordinates"""
        # Symmetric-to-prenek vertex ordering
        isym2pre = [0, 1, 3, 2, 4, 5, 7, 6]  # 0-based
        
        rgroup = np.float64(0.0)
        
        for iel in range(self.mesh.num_elem):
            # Write group number
            self.file.write(rgroup.tobytes())
            
            # Extract vertices
            if self.mesh.num_dim == 3:
                xx = np.zeros(8)
                yy = np.zeros(8)
                zz = np.zeros(8)
                
                l = 0
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            li = isym2pre[l]
                            xx[li] = self.mesh.xm1[i*2, j*2, k*2, iel]
                            yy[li] = self.mesh.ym1[i*2, j*2, k*2, iel]
                            zz[li] = self.mesh.zm1[i*2, j*2, k*2, iel]
                            l += 1
                
                # Write coordinates
                self.file.write(xx.astype(np.float64).tobytes())
                self.file.write(yy.astype(np.float64).tobytes())
                self.file.write(zz.astype(np.float64).tobytes())
                
            else:  # 2D
                xx = np.zeros(4)
                yy = np.zeros(4)
                
                l = 0
                for j in range(2):
                    for i in range(2):
                        li = isym2pre[l]
                        xx[li] = self.mesh.xm1[i*2, j*2, 0, iel]
                        yy[li] = self.mesh.ym1[i*2, j*2, 0, iel]
                        l += 1
                
                # Write coordinates
                self.file.write(xx.astype(np.float64).tobytes())
                self.file.write(yy.astype(np.float64).tobytes())
    
    def _write_curve(self):
        """Write curved side data"""
        # Generate midside data
        self._gen_rea_midside()
        
        # Count curved sides
        nedge = 4 + 8 * (self.mesh.num_dim - 2)
        ncurv = 0
        
        for iel in range(self.mesh.num_elem):
            for iedge in range(nedge):
                if self.mesh.ccurve[iedge, iel] != ' ':
                    ncurv += 1
        
        # Write curve count
        rcurve = np.float64(ncurv)
        self.file.write(rcurve.tobytes())
        
        # Write curve data
        for iel in range(self.mesh.num_elem):
            for iedge in range(nedge):
                if self.mesh.ccurve[iedge, iel] != ' ':
                    buf = np.zeros(16, dtype=np.float64)
                    buf[0] = float(iel)
                    buf[1] = float(iedge)
                    buf[2:7] = self.mesh.curve[:5, iedge, iel]
                    
                    # Write buffer
                    self.file.write(buf[:8].tobytes())
                    
                    # Write character
                    cc = self.mesh.ccurve[iedge, iel].encode('ascii')
                    cc_buf = cc.ljust(8, b' ')
                    self.file.write(cc_buf)
    
    def _write_bc(self):
        """Write boundary condition data"""
        nface = 2 * self.mesh.num_dim
        
        # Count BCs for velocity field
        nbc = 0
        for iel in range(self.mesh.eftot):
            for ifc in range(nface):
                if self.mesh.cbc[ifc, iel] != '   ':
                    nbc += 1
        
        # Write BC count
        rbc = np.float64(nbc)
        self.file.write(rbc.tobytes())
        
        logger.info(f"Velocity boundary faces: {nbc}")
        
        # Write velocity BCs
        for iel in range(self.mesh.eftot):
            for ifc in range(nface):
                if self.mesh.cbc[ifc, iel] != '   ':
                    buf = np.zeros(16, dtype=np.float64)
                    buf[0] = float(iel)
                    buf[1] = float(ifc)
                    buf[2:7] = self.mesh.bc[:5, ifc, iel]
                    
                    # Handle large element numbers
                    if self.mesh.eftot >= 1000000:
                        ibc = int(self.mesh.bc[0, ifc, iel])
                        buf[2] = float(ibc)
                    
                    # Write buffer
                    self.file.write(buf[:8].tobytes())
                    
                    # Write BC string
                    ch3 = self.mesh.cbc[ifc, iel].encode('ascii')
                    ch3_buf = ch3.ljust(8, b' ')
                    self.file.write(ch3_buf)
        
        # Write thermal BCs if needed
        if self.mesh.num_elem != self.mesh.eftot:
            nbc = 0
            for iel in range(self.mesh.num_elem):
                for ifc in range(nface):
                    if self.mesh.cbc[ifc, iel] != '   ':
                        nbc += 1
            
            rbc = np.float64(nbc)
            self.file.write(rbc.tobytes())
            
            logger.info(f"Thermal boundary faces: {nbc}")
            
            for iel in range(self.mesh.num_elem):
                for ifc in range(nface):
                    if self.mesh.cbc[ifc, iel] != '   ':
                        buf = np.zeros(16, dtype=np.float64)
                        buf[0] = float(iel)
                        buf[1] = float(ifc)
                        buf[2:7] = self.mesh.bc[:5, ifc, iel]
                        
                        if self.mesh.num_elem >= 1000000:
                            ibc = int(self.mesh.bc[0, ifc, iel])
                            buf[2] = float(ibc)
                        
                        self.file.write(buf[:8].tobytes())
                        
                        ch3 = self.mesh.cbc[ifc, iel].encode('ascii')
                        ch3_buf = ch3.ljust(8, b' ')
                        self.file.write(ch3_buf)
    
    def _gen_rea_midside(self):
        """Generate midside node data for curved elements"""
        tol = 1e-4
        tol2 = tol**2
        nedge = 4 + 8 * (self.mesh.num_dim - 2)
        
        for e in range(self.mesh.num_elem):
            # Map to 3x3x3 array
            x3 = self._map2reg(self.mesh.xm1[:, :, :, e])
            y3 = self._map2reg(self.mesh.ym1[:, :, :, e])
            if self.mesh.num_dim == 3:
                z3 = self._map2reg(self.mesh.zm1[:, :, :, e])
            
            # Handle spherical curved faces
            ccrve = self.mesh.ccurve[:, e].copy()
            if self.mesh.num_dim == 3:
                if self.mesh.ccurve[4, e] == 's':
                    ccrve[:4] = 's'
                    ccrve[4] = ' '
                if self.mesh.ccurve[5, e] == 's':
                    ccrve[4:8] = 's'
            
            # Check edges for curvature
            e3 = np.array(self.converter.e3 if hasattr(self, 'converter') else [
                [0, 1, 2], [2, 5, 8], [8, 7, 6], [6, 3, 0],
                [18, 19, 20], [20, 23, 26], [26, 25, 24], [24, 21, 18],
                [0, 9, 18], [2, 11, 20], [8, 17, 26], [6, 15, 24]
            ])
            
            for i in range(nedge):
                if ccrve[i] == ' ':
                    xyz = np.zeros((3, 3))
                    for j in range(3):
                        xyz[0, j] = x3[e3[i, j]]
                        xyz[1, j] = y3[e3[i, j]]
                        if self.mesh.num_dim == 3:
                            xyz[2, j] = z3[e3[i, j]]
                    
                    # Check for curvature
                    xmid = 0.5 * (xyz[:, 0] + xyz[:, 2])
                    h = np.sum((xyz[:, 1] - xmid)**2)
                    length = np.sum((xyz[:, 2] - xyz[:, 0])**2)
                    
                    if h > tol2 * length:
                        self.mesh.ccurve[i, e] = 'm'
                        self.mesh.curve[:self.mesh.num_dim, i, e] = xyz[:self.mesh.num_dim, 1]
    
    def _map2reg(self, u):
        """Map field to regular 3x3x3 array"""
        n = 3
        m = 3
        
        if self.mesh.num_dim == 2:
            # Generate interpolation weights
            z_gll = self._zwgll(m)
            z_uni = np.linspace(-1, 1, n)
            
            j, jt = self._gen_int_gz(z_uni, z_gll)
            
            # Interpolate
            ur = np.zeros((n, n))
            w = j @ u @ jt.T
            ur = w
            
        else:  # 3D
            z_gll = self._zwgll(m)
            z_uni = np.linspace(-1, 1, n)
            
            j, jt = self._gen_int_gz(z_uni, z_gll)
            
            # Interpolate in 3D
            ur = np.zeros((n, n, n))
            
            # First direction
            v = np.zeros((n, m, m))
            for k in range(m):
                for j_idx in range(m):
                    v[:, j_idx, k] = j @ u[:, j_idx, k]
            
            # Second direction
            w = np.zeros((n, n, m))
            for k in range(m):
                w[:, :, k] = v[:, :, k] @ jt.T
            
            # Third direction
            for i in range(n):
                for j_idx in range(n):
                    ur[i, j_idx, :] = jt @ w[i, j_idx, :]
        
        return ur.flatten()
    
    def _zwgll(self, n):
        """Gauss-Lobatto-Legendre points"""
        if n == 3:
            return np.array([-1.0, 0.0, 1.0])
        else:
            # For general case, would need to compute GLL points
            raise NotImplementedError("GLL points for n != 3 not implemented")
    
    def _gen_int_gz(self, g, z):
        """Generate interpolation matrix from z to g points"""
        n = len(g)
        m = len(z)
        
        jt = np.zeros((m, n))
        
        # Lagrange interpolation weights
        for i in range(n):
            weights = self._fd_weights_full(g[i], z, m-1)
            jt[:, i] = weights
        
        j = jt.T
        
        return j, jt
    
    def _fd_weights_full(self, xx, x, n):
        """Fornberg's algorithm for finite difference weights"""
        c = np.zeros((n+1, n+1))
        c[0, 0] = 1.0
        c1 = 1.0
        c4 = x[0] - xx
        
        for i in range(1, n+1):
            mn = min(i, n)
            c2 = 1.0
            c5 = c4
            c4 = x[i] - xx
            
            for j in range(i):
                c3 = x[i] - x[j]
                c2 = c2 * c3
                
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i-1, k-1] - c5 * c[i-1, k]) / c2
                
                c[i, 0] = -c1 * c5 * c[i-1, 0] / c2
                
                for k in range(mn, 0, -1):
                    c[j, k] = (c4 * c[j, k] - k * c[j, k-1]) / c3
                
                c[j, 0] = c4 * c[j, 0] / c3
            
            c1 = c2
        
        return c[:, 0]


def main():
    """Main program"""
    logger.info("GMSH to NEK5000 Mesh Converter")
    logger.info("==============================")
    
    # Get mesh dimension
    while True:
        try:
            dim = int(input("Enter mesh dimension (2 or 3): "))
            if dim in [2, 3]:
                break
            else:
                logger.error("Please enter 2 or 3")
        except ValueError:
            logger.error("Invalid input")
    
    # Initialize reader and converter
    reader = GmshReader()
    converter = MeshConverter()
    
    # Read fluid mesh
    fluid_file = input("Input fluid .msh file name: ").strip()
    if not fluid_file.endswith('.msh'):
        fluid_file += '.msh'
    
    # Check file format
    version, file_type = reader.read_file_header(fluid_file)
    
    # Pre-read fluid mesh
    logger.info(f"Reading mesh file {fluid_file}")
    
    if dim == 2:
        if file_type == 0:  # ASCII
            mesh = reader.read_2d_ascii(fluid_file, preread=True)
        else:  # Binary
            mesh = reader.read_2d_binary(fluid_file, preread=True)
    else:  # 3D
        if file_type == 0:  # ASCII
            mesh = reader.read_3d_ascii(fluid_file, preread=True)
        else:  # Binary
            mesh = reader.read_3d_binary(fluid_file, preread=True)
    
    eftot = mesh.num_elem
    
    if dim == 2:
        logger.info(f"Total fluid quad number: {eftot}")
    else:
        logger.info(f"Total fluid hex number: {eftot}")
    
    # Check for solid mesh
    ifsolid = int(input("Do you have solid mesh? (0 for no, 1 for yes): "))
    
    total_elem = eftot
    
    if ifsolid == 1:
        solid_file = input("Input solid .msh file name: ").strip()
        if not solid_file.endswith('.msh'):
            solid_file += '.msh'
        
        # Check solid file format
        version_s, file_type_s = reader.read_file_header(solid_file)
        
        logger.info(f"Reading mesh file {solid_file}")
        
        if dim == 2:
            if file_type_s == 0:  # ASCII
                mesh_solid = reader.read_2d_ascii(solid_file, preread=True)
            else:  # Binary
                mesh_solid = reader.read_2d_binary(solid_file, preread=True)
        else:  # 3D
            if file_type_s == 0:  # ASCII
                mesh_solid = reader.read_3d_ascii(solid_file, preread=True)
            else:  # Binary
                mesh_solid = reader.read_3d_binary(solid_file, preread=True)
        
        total_elem = eftot + mesh_solid.num_elem
        
        if dim == 2:
            logger.info(f"Total quad number: {total_elem}")
        else:
            logger.info(f"Total hex number: {total_elem}")
    
    # Allocate arrays for all elements
    mesh_full = MeshData()
    mesh_full.num_dim = dim
    mesh_full.num_elem = total_elem
    mesh_full.eftot = eftot
    
    # Allocate coordinate arrays
    mesh_full.xm1 = np.zeros((3, 3, 3, total_elem))
    mesh_full.ym1 = np.zeros((3, 3, 3, total_elem))
    mesh_full.zm1 = np.zeros((3, 3, 3, total_elem))
    
    # Allocate curve arrays
    nedge = 4 + 8 * (dim - 2)
    mesh_full.ccurve = np.full((nedge, total_elem), ' ', dtype='U1')
    mesh_full.curve = np.zeros((2 * dim, 12, total_elem))
    
    # Allocate BC arrays
    nface = 2 * dim
    mesh_full.cbc = np.full((nface, total_elem), '   ', dtype='U3')
    mesh_full.bc = np.zeros((5, nface, total_elem))
    
    if dim == 2:
        mesh_full.quad_line_array = np.zeros((4, total_elem), dtype=int)
        mesh_full.r_or_l = np.zeros(total_elem, dtype=int)
    else:
        mesh_full.hex_face_array = np.zeros((6, total_elem), dtype=int)
    
    # Read and convert fluid mesh
    logger.info("\nProcessing fluid mesh...")
    
    if dim == 2:
        if file_type == 0:  # ASCII
            mesh = reader.read_2d_ascii(fluid_file, preread=False)
        else:  # Binary
            mesh = reader.read_2d_binary(fluid_file, preread=False)
    else:  # 3D
        if file_type == 0:  # ASCII
            mesh = reader.read_3d_ascii(fluid_file, preread=False)
        else:  # Binary
            mesh = reader.read_3d_binary(fluid_file, preread=False)
    
    # Copy data to full mesh
    mesh_full.node_xyz = mesh.node_xyz
    mesh_full.node_line = mesh.node_line if hasattr(mesh, 'node_line') else None
    mesh_full.node_quad = mesh.node_quad
    mesh_full.node_hex = mesh.node_hex if hasattr(mesh, 'node_hex') else None
    mesh_full.line_array = mesh.line_array if hasattr(mesh, 'line_array') else None
    mesh_full.quad_array = mesh.quad_array
    mesh_full.hex_array = mesh.hex_array if hasattr(mesh, 'hex_array') else None
    mesh_full.bc_number = mesh.bc_number
    mesh_full.bc_id = mesh.bc_id
    mesh_full.bc_char = mesh.bc_char
    mesh_full.total_line = mesh.total_line
    mesh_full.total_quad = mesh.total_quad
    mesh_full.total_hex = mesh.total_hex if hasattr(mesh, 'total_hex') else 0
    
    # Convert fluid elements
    if dim == 2:
        converter.convert_2d(mesh_full, start_quad=0)
    else:
        converter.convert_3d(mesh_full, start_hex=0)
    
    # Print boundary info
    logger.info("\n******************************************************")
    logger.info("Fluid mesh boundary info summary")
    logger.info("BoundaryName     BoundaryID")
    for i in range(mesh.bc_number):
        logger.info(f"{mesh.bc_char[i]}    {mesh.bc_id[0, i]}")
    logger.info("******************************************************")
    
    # Process solid mesh if present
    if ifsolid == 1:
        logger.info("\nProcessing solid mesh...")
        
        if dim == 2:
            if file_type_s == 0:  # ASCII
                mesh_solid = reader.read_2d_ascii(solid_file, preread=False)
            else:  # Binary
                mesh_solid = reader.read_2d_binary(solid_file, preread=False)
        else:  # 3D
            if file_type_s == 0:  # ASCII
                mesh_solid = reader.read_3d_ascii(solid_file, preread=False)
            else:  # Binary
                mesh_solid = reader.read_3d_binary(solid_file, preread=False)
        
        # Update full mesh with solid data
        # Note: This is simplified - in practice would need to merge node data properly
        
        # Convert solid elements
        if dim == 2:
            converter.convert_2d(mesh_solid, start_quad=eftot)
            # Copy converted data
            for i in range(mesh_solid.num_elem):
                mesh_full.xm1[:, :, :, eftot + i] = mesh_solid.xm1[:, :, :, i]
                mesh_full.ym1[:, :, :, eftot + i] = mesh_solid.ym1[:, :, :, i]
                mesh_full.quad_line_array[:, eftot + i] = mesh_solid.quad_line_array[:, i]
                mesh_full.r_or_l[eftot + i] = mesh_solid.r_or_l[i]
                mesh_full.cbc[:, eftot + i] = mesh_solid.cbc[:, i]
                mesh_full.bc[:, :, eftot + i] = mesh_solid.bc[:, :, i]
        else:
            converter.convert_3d(mesh_solid, start_hex=eftot)
            # Copy converted data
            for i in range(mesh_solid.num_elem):
                mesh_full.xm1[:, :, :, eftot + i] = mesh_solid.xm1[:, :, :, i]
                mesh_full.ym1[:, :, :, eftot + i] = mesh_solid.ym1[:, :, :, i]
                mesh_full.zm1[:, :, :, eftot + i] = mesh_solid.zm1[:, :, :, i]
                mesh_full.hex_face_array[:, eftot + i] = mesh_solid.hex_face_array[:, i]
                mesh_full.cbc[:, eftot + i] = mesh_solid.cbc[:, i]
                mesh_full.bc[:, :, eftot + i] = mesh_solid.bc[:, :, i]
        
        # Print solid boundary info
        logger.info("\n******************************************************")
        logger.info("Solid mesh boundary info summary")
        logger.info("BoundaryName     BoundaryID")
        for i in range(mesh_solid.bc_number):
            logger.info(f"{mesh_solid.bc_char[i]}    {mesh_solid.bc_id[0, i]}")
        logger.info("******************************************************")
    
    # Check for left-handed elements (3D only)
    if dim == 3:
        ne_nrh = check_right_hand_elements(mesh_full)
        if ne_nrh > 0:
            logger.warning(f"Found {ne_nrh} left-handed elements")
            fix_left_hand_elements_3d(mesh_full)
    
    # Set periodic boundary conditions
    converter.set_periodicity(mesh_full, field_type=1)
    if eftot != total_elem:
        converter.set_periodicity(mesh_full, field_type=2)
    
    # Write re2 file
    re2_name = input("\nPlease give re2 file name: ").strip()
    if not re2_name.endswith('.re2'):
        re2_name += '.re2'
    
    # Add converter reference to writer
    writer = Re2Writer(mesh_full)
    writer.converter = converter
    writer.write(re2_name)
    
    logger.info("\nConversion complete!")


def check_right_hand_elements(mesh: MeshData) -> int:
    """Check for non-right-handed elements in 3D mesh"""
    ne_nrh = 0
    
    for iel in range(mesh.num_elem):
        # Get corner vertices
        x = np.array([mesh.xm1[0, 0, 0, iel], mesh.xm1[2, 0, 0, iel],
                      mesh.xm1[2, 2, 0, iel], mesh.xm1[0, 2, 0, iel],
                      mesh.xm1[0, 0, 2, iel], mesh.xm1[2, 0, 2, iel],
                      mesh.xm1[2, 2, 2, iel], mesh.xm1[0, 2, 2, iel]])
        
        y = np.array([mesh.ym1[0, 0, 0, iel], mesh.ym1[2, 0, 0, iel],
                      mesh.ym1[2, 2, 0, iel], mesh.ym1[0, 2, 0, iel],
                      mesh.ym1[0, 0, 2, iel], mesh.ym1[2, 0, 2, iel],
                      mesh.ym1[2, 2, 2, iel], mesh.ym1[0, 2, 2, iel]])
        
        z = np.array([mesh.zm1[0, 0, 0, iel], mesh.zm1[2, 0, 0, iel],
                      mesh.zm1[2, 2, 0, iel], mesh.zm1[0, 2, 0, iel],
                      mesh.zm1[0, 0, 2, iel], mesh.zm1[2, 0, 2, iel],
                      mesh.zm1[2, 2, 2, iel], mesh.zm1[0, 2, 2, iel]])
        
        # Compute volume using triple product
        v1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
        v2 = np.array([x[3] - x[0], y[3] - y[0], z[3] - z[0]])
        v3 = np.array([x[4] - x[0], y[4] - y[0], z[4] - z[0]])
        
        volume = np.dot(v1, np.cross(v2, v3))
        
        if volume < 0:
            ne_nrh += 1
    
    return ne_nrh


def fix_left_hand_elements_3d(mesh: MeshData):
    """Fix left-handed elements by swapping nodes"""
    for iel in range(mesh.num_elem):
        # Get corner vertices
        x = np.array([mesh.xm1[0, 0, 0, iel], mesh.xm1[2, 0, 0, iel],
                      mesh.xm1[2, 2, 0, iel], mesh.xm1[0, 2, 0, iel],
                      mesh.xm1[0, 0, 2, iel], mesh.xm1[2, 0, 2, iel],
                      mesh.xm1[2, 2, 2, iel], mesh.xm1[0, 2, 2, iel]])
        
        y = np.array([mesh.ym1[0, 0, 0, iel], mesh.ym1[2, 0, 0, iel],
                      mesh.ym1[2, 2, 0, iel], mesh.ym1[0, 2, 0, iel],
                      mesh.ym1[0, 0, 2, iel], mesh.ym1[2, 0, 2, iel],
                      mesh.ym1[2, 2, 2, iel], mesh.ym1[0, 2, 2, iel]])
        
        z = np.array([mesh.zm1[0, 0, 0, iel], mesh.zm1[2, 0, 0, iel],
                      mesh.zm1[2, 2, 0, iel], mesh.zm1[0, 2, 0, iel],
                      mesh.zm1[0, 0, 2, iel], mesh.zm1[2, 0, 2, iel],
                      mesh.zm1[2, 2, 2, iel], mesh.zm1[0, 2, 2, iel]])
        
        # Compute volume
        v1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
        v2 = np.array([x[3] - x[0], y[3] - y[0], z[3] - z[0]])
        v3 = np.array([x[4] - x[0], y[4] - y[0], z[4] - z[0]])
        
        volume = np.dot(v1, np.cross(v2, v3))
        
        if volume < 0:
            # Swap face 5 and 6 (top and bottom)
            temp_x = mesh.xm1[:, :, 0, iel].copy()
            temp_y = mesh.ym1[:, :, 0, iel].copy()
            temp_z = mesh.zm1[:, :, 0, iel].copy()
            
            mesh.xm1[:, :, 0, iel] = mesh.xm1[:, :, 2, iel]
            mesh.ym1[:, :, 0, iel] = mesh.ym1[:, :, 2, iel]
            mesh.zm1[:, :, 0, iel] = mesh.zm1[:, :, 2, iel]
            
            mesh.xm1[:, :, 2, iel] = temp_x
            mesh.ym1[:, :, 2, iel] = temp_y
            mesh.zm1[:, :, 2, iel] = temp_z
            
            # Also swap boundary conditions
            temp_cbc = mesh.cbc[4, iel]
            mesh.cbc[4, iel] = mesh.cbc[5, iel]
            mesh.cbc[5, iel] = temp_cbc
            
            temp_bc = mesh.bc[:, 4, iel].copy()
            mesh.bc[:, 4, iel] = mesh.bc[:, 5, iel]
            mesh.bc[:, 5, iel] = temp_bc


if __name__ == "__main__":
    main()