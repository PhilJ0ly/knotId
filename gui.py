# Philippe Joly 22-12-24

import os
import numpy as np
import networkx as nx
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
from pyknotid.spacecurves import Knot

import tkinter as tk
from tkinter import Label, Button, Radiobutton, IntVar

from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage import morphology

class KnotId:    
    def segment(img):
        img_gray = rgb2gray(img)

        elevation_map = sobel(img_gray)

        markers = np.zeros_like(img_gray)
        markers[img_gray<30/255]=1
        markers[img_gray>150/255]=2
        markers = markers.astype(np.int32)

        bw = watershed(elevation_map, markers)-1
        # bw = blur(bw, 10, 16)
        return bw

    def skeletonize(seg):
        skeleton = morphology.skeletonize(seg)
        return skeleton

    def get_graph(skeleton):
        G = nx.Graph()
        rows, cols = skeleton.shape
        for r in range(rows):
            for c in range(cols):
                if skeleton[r, c]:
                    G.add_node((r, c))

        # Add edges between adjacent pixels
        slants = [(-1, -1),(-1, 1),(1, -1),(1, 1)]
        
        directions = [(-1, 0),(0, -1),(0, 1),(1, 0)]

        for node in G.nodes():
            r, c = node
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (nr, nc) in G.nodes():
                    G.add_edge((r, c), (nr, nc))
            
            if G.degree((r,c))<2:
                for dr, dc in slants:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in G.nodes():
                        G.add_edge((r, c), (nr, nc))
    
        return G
    
    def get_clean_graph(G, scale=1):
        # Remove small loops
        def search_loop(G, init_node, prev_node, node, count, limit):
            if node == init_node:
                G.remove_edge(prev_node, node)
                return True
            
            flag = False 

            if count < limit:
                for neighbour in G.neighbors(node):
                    if neighbour != prev_node:
                        flag = search_loop(G, init_node, node, neighbour, count+1, limit)

                    if flag:
                        break

            return flag
        
        def remove_loops(G, limit):
            for i, node in enumerate(G.nodes):
                if G.degree(node) > 2:
                    for neighbour in G.neighbors(node):
                        flag = False
                        if neighbour !=list(G.nodes)[i-1]:
                            flag = search_loop(G, node, node, neighbour, 0, limit)

                        if flag:
                            break

        # remove unlooped branches
        def remove_unconnected(G, limit):
            removable = []

            for node in G.nodes:
                if G.degree(node) < 2:
                    rmv = []
                    flag = search_branch(G, None, node, rmv, 0, limit)
                    if flag:
                        removable.extend(rmv)
                        
            removable = list(set(removable))
            for node in removable:
                G.remove_node(node)

        def search_branch(G, prev_node, node, removable, count, limit):
            if G.degree(node) > 2:
                return True
            
            if count >= limit:
                return False
            
            removable.append(node)
            for neighbour in G.neighbors(node):
                if neighbour != prev_node:
                    return search_branch(G, node, neighbour, removable, count+1, limit)
                
            return True
        
        # connect close intersections
        def connect_int(G, limit):
            ints = []
            connections = []
            removes = []
            for i, node in enumerate(G.nodes):
                if G.degree(node) == 3:
                    for neighbour in G.neighbors(node):
                        if neighbour != list(G.nodes)[i-1]:
                            inter, removable = search_int(G, node, neighbour, 0, limit, [])
                            if inter and (inter, node) not in connections:
                                ints.extend([node,inter])
                                connections.append((node, inter))
                                removes.extend(removable)
            
            removes = list(set(removes))
            for node in removes:
                G.remove_node(node)
            
            new_ints = []
            connections = list(set(connections))
            for int1, int2 in connections:
                r, c = (int1[0]+int2[0])//2, (int1[1]+int2[1])//2
                G.add_node((r, c))

                for neighbour in list(G.neighbors(int1))+list(G.neighbors(int2)):
                    if neighbour != (r,c):
                        G.add_edge((r,c), neighbour)
                new_ints.append((r,c))
            
            ints = list(set(ints))
            for node in ints:
                if node not in new_ints:
                    G.remove_node(node)
        
        def search_int(G, prev_node, node, count, limit, removable):
            if G.degree(node) == 3:
                return node, removable
            
            if count > limit:
                return None, None

            for neighbour in G.neighbors(node):
                if neighbour != prev_node:
                    removable.append(node)
                    return search_int(G, node, neighbour, count+1, limit, removable)  

            return None, None  
        
        # tying up loose ends
        def dis(n1, n2):
            return np.sqrt((n2[0]-n1[0])**2+(n2[1]-n1[1])**2)

        def tying_loose_ends(G, limit):
            threes = []
            singles = []

            for node in G.nodes:
                if G.degree(node) == 3:
                    threes.append(node)
                if G.degree(node) == 1:
                    singles.append(node)

            singles = connect_singles(G, singles, limit)
            connect_1_3(G, singles, threes, limit)

        def connect_1_3(G, singles, threes, limit):
            l3 = len(threes)
            l1 = len(singles)
            if l1 == 0 or l3 == 0:
                return threes+singles
            
            distances = []
            for i in range(l1):
                dist = []
                for j in range(l3):
                    dist.append((j+l1, dis(singles[i], threes[j])))
                dist = sorted(dist, key=lambda x: x[1])
                distances.append(dist)
            for i in range(l3):
                dist = []
                for j in range(l1):
                    dist.append((j, dis(threes[i], singles[j])))
                dist = sorted(dist, key=lambda x: x[1])
                distances.append(dist)

            closests, _, single = match_closest([], distances, singles+threes)

            for el in closests:
                if el[2] < limit:
                    G.add_edge(el[0], el[1])
            
            return single

        def connect_singles(G, singles, limit):
            l1 = len(singles)
            if l1 < 2:
                return singles
            
            distances = []
            for i in range(l1-1):
                dist = []
                for j in range(i+1,l1):
                    dist.append((j, dis(singles[i], singles[j])))
                dist = sorted(dist, key=lambda x: x[1])
                distances.append(dist)

            closests, _, single = match_closest([], distances, singles)
            
            for el in closests:
                if el[-1] < limit:
                    G.add_edge(el[0], el[1])
            
            return single                     

        def match_closest(closests, distances, singles):
            if len(singles) < 2:
                return closests, distances, singles
            
            closest = []
            closest_idx = []
            for i in range(len(singles)-1):
                j = distances[distances[i][0][0]][0][0]
                if j == i:
                    if [singles[j],singles[i], distances[j][0][1]] not in closest:
                        closest.append([singles[i],singles[distances[i][0][0]], distances[i][0][1]])
                        closest_idx.extend([i,distances[i][0][0]])
            
            closests.extend(closest)
            closest_idx = sorted(list(set(closest_idx)), reverse=True)
            new_singles = singles.copy()

            for idx in closest_idx:
                new_singles.pop(idx)
                distances.pop(idx)
            
            for i in range(len(distances)):
                distances[i] = [x for x in distances[i] if x[0] not in closest_idx]

            return match_closest(closests, distances, new_singles)  

        
        structs = list(nx.connected_components(G))
        largest_struct = max(structs, key=len)
        G = G.subgraph(largest_struct).copy()

        # scale = 1 # scaling factor to adjust for picture resolution (1 for example_1.png)

        remove_loops(G, scale*5)
        for i in range(5):
            remove_unconnected(G, scale*2*(i+1))
        connect_int(G, scale*15)
        tying_loose_ends(G, scale*30)
        connect_int(G, scale*25)
        remove_unconnected(G, scale*25)

        return G

    def get_graph_3d(G):
        for node in list(G.nodes):
            r, c = node
            G = nx.relabel_nodes(G, {node: (r, c, 0)})

        edges = list(G.edges)
        G.remove_edges_from(edges)
        for edge in edges:
            node1, node2 = edge
            G.add_edge((node1[0], node1[1], 0), (node2[0], node2[1], 0))
        
        return G
    
    def check_graph(G):
        errors = []
        for node in G.nodes:
            deg = G.degree(node)
            if deg != 2 and deg != 4:
                errors.append((deg, node))

        err_num = len(errors)
        if err_num>0:
            print(err_num, "errors in the graphing")
            print(errors)
            return 1
        else:
            print("No apparent Errors in Graph")
            return 0

    def get_crossings(G, img):    
        def opposite_pair(n):
            n0 = n[0]
            
            for i in range(2):
                ni = n[i+1]
                if n0[0]-ni[0] != 0:
                    m = float(n0[1]-ni[1])/float(n0[0]-ni[0])
                    b = n0[1]-m*n0[0]

                    over = 0
                    for j in range(3):
                        if i == j:
                            pass

                        y_pred = m*n[j+1][0]+b
                        
                        if n[j+1][1] > y_pred:
                            over += 1
                else:
                    over = 0
                    for j in range(3):
                        if i == j:
                            pass
                        if n[j+1][0] > n0[0]:
                            over += 1
                
                if over == 1:
                    p1 = [n0, n[i+1]]
                    p2 = [el for el in n if el not in p1]
                    return (p1, p2)
            
            return ([n0, n[-1]], [n[1], n[2]])

        def show_crossings(node, pairs, img, r, c):
            neighbors = pairs[0] + pairs[1]

            # Calculate the bounds of the square to fit all neighbors
            x_coords = [node[0]] + [neighbor[0] for neighbor in neighbors]
            y_coords = [node[1]] + [neighbor[1] for neighbor in neighbors]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            square_size = max(c, max(max_x - min_x, max_y - min_y) + 2 * r)

            square_left = int(max(0, node[0] - square_size))
            square_right = int(min(img.size[1], node[0] + square_size))
            square_bottom = int(max(0, node[1] - square_size))
            square_top = int(min(img.size[0], node[1] + square_size))

            # Crop the image to the square boundaries
            img_array = np.array(img)
            img_cropped = img_array[square_left:square_right, square_bottom:square_top]
            img = Image.fromarray(img_cropped)

            draw = ImageDraw.Draw(img)
            offset_x, offset_y = square_left, square_bottom

            # Draw neighbors
            for n in range(2):
                draw.ellipse([
                    (pairs[0][n][0] - r - offset_x, pairs[0][n][1] - r - offset_y),
                    (pairs[0][n][0] + r - offset_x, pairs[0][n][1] + r - offset_y)
                ], fill="red")

                draw.ellipse([
                    (pairs[1][n][0] - r - offset_x, pairs[1][n][1] - r - offset_y),
                    (pairs[1][n][0] + r - offset_x, pairs[1][n][1] + r - offset_y)
                ], fill="blue")

            draw.ellipse([
                (node[0] - r - offset_x, node[1] - r - offset_y),
                (node[0] + r - offset_x, node[1] + r - offset_y)
            ], fill="black")

            return img

        crossings = []
        for node in G.nodes:
            if G.degree(node) == 4:
                neighbors = list(G.neighbors(node))
                p1, p2 = opposite_pair(neighbors)
                crossings.append([
                    node, 
                    (p1, p2), 
                    show_crossings(node, (p1, p2), img, 1, 15),
                    0
                ])

        return crossings
    
    def get_over_under(crossings):
        # function using gui to determine which pair is up
        # blue_up -> crossings[i][-1] = 1
        blue_ups = []

        def callback(collected_states):
            nonlocal blue_ups
            blue_ups = collected_states

        root = tk.Tk()
        viewer = ImageViewer(root, [cross[2] for cross in crossings], callback)
        root.mainloop()

        for i in range(len(crossings)):
            crossings[i][-1] = blue_ups[i]
        
        return crossings

    def graph_3d_crossings(G, crossings):
        for i, cross in enumerate(crossings):
            top_pair = cross[1][cross[-1]]

            new_nodes = [(),()]
            for i in range(2):
                G.remove_edge(top_pair[i], cross[0])
                new_nodes[i] = (top_pair[i][0], top_pair[i][1], 1)
                G.add_node(new_nodes[i])

                for n in list(G.neighbors(top_pair[i])):
                    G.add_edge(new_nodes[i], n)
                
                G.remove_node(top_pair[i])

            G.add_edge(new_nodes[0], new_nodes[1])

    def get_graph_to_array(G):
        start_node = None
        for node in G.nodes:
            start_node = node
            if G.degree(node) == 1:
                break

        ordered_points = []
        visited = set()
        current_node = start_node
        prev_node = None

        while current_node is not None:
            ordered_points.append(current_node)
            visited.add(current_node)
            neighbors = list(G.neighbors(current_node))
            next_node = None
            for neighbor in neighbors:
                if neighbor != prev_node and neighbor not in visited:
                    next_node = neighbor
                    break
            prev_node = current_node
            current_node = next_node

        points_array = np.array(ordered_points)
        return points_array

    def get_knotid(pts):
        k = Knot(pts)
        knotid = k.identify()
        return knotid
    
    def knot_identify(image):
        seg = KnotId.segment(image)
        skeleton = KnotId.skeletonize(seg)

        G = KnotId.get_graph(skeleton)
        G = KnotId.get_clean_graph(G)
        G = KnotId.get_graph_3d(G)

        crossings = KnotId.get_crossings(G, image)
        crossings = KnotId.get_over_under(crossings)

        KnotId.graph_3d_crossings(G, crossings)
        pts = KnotId.get_graph_to_array(G)

        knotid = KnotId.get_knotid(pts)

        return knotid
    
    def group_knots_identify(imgs):
        graphs = []
        crosses = []
        
        for image in imgs:
            seg = KnotId.segment(image)
            skeleton = KnotId.skeletonize(seg)

            G = KnotId.get_graph(skeleton)
            G = KnotId.get_clean_graph(G)
            G = KnotId.get_graph_3d(G)

            crossings = KnotId.get_crossings(G, image)

            graphs.append(G)
            crosses.append(crossings)

        holy_cross = []
        shapes = []

        for crossings in crosses:
            shapes.append(len(crossings))
            holy_cross.extend(crossings)
        
        holy_cross = KnotId.get_over_under(holy_cross)
        crosses = []
        idx = 0

        for sz in shapes:
            crosses.append(holy_cross[idx:idx+sz])
            idx += sz

        knotIds = []
        for i in range(len(graphs)):
            G = graphs[i]
            crossings = crosses[i]

            KnotId.graph_3d_crossings(G, crossings)
            pts = KnotId.get_graph_to_array(G)

            knotid = KnotId.get_knotid(pts)

            knotIds.append(knotid)
        
        return knotIds


class ImageViewer:
    def __init__(self, root, pil_images, states_callback):
        self.root = root
        self.images = pil_images
        self.current_index = 0
        self.states_callback = states_callback

        # State dictionary to keep track of radio button values for each image
        self.radio_states = {i: IntVar(value=0) for i in range(len(self.images))}

        if not self.images:
            raise ValueError("No images provided.")

        # Counter label
        self.counter_label = Label(root, text="", font=("Arial", 14))
        self.counter_label.pack()

        # Image display label
        self.image_label = Label(root)
        self.image_label.pack()

        # Navigation buttons
        self.prev_button = Button(root, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = Button(root, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # Radio button for Blue/Red state
        self.radio_on = Radiobutton(root, text="Blue", variable=self.radio_states[self.current_index], value=1, command=self.update_radio_state)
        self.radio_off = Radiobutton(root, text="Red", variable=self.radio_states[self.current_index], value=0, command=self.update_radio_state)
        self.radio_on.pack(side=tk.LEFT, padx=10, pady=10)
        self.radio_off.pack(side=tk.LEFT, padx=10, pady=10)

        # Submit button
        self.submit_button = Button(root, text="Submit", command=self.submit_states)
        self.submit_button.pack(side=tk.BOTTOM, pady=20)

        # Keyboard bindings
        root.bind('<Left>', lambda event: self.show_previous_image())
        root.bind('<Right>', lambda event: self.show_next_image())
        root.bind('<Up>', lambda event: self.toggle_radio_state())
        root.bind('<Return>', lambda event: self.submit_states())

        # Load the first image
        self.update_image()

    def update_image(self):
        img = self.images[self.current_index]
        img = img.resize((500, 500), Image.Resampling.LANCZOS)  # Resize image to fit in the window
        self.photo = ImageTk.PhotoImage(img)

        self.image_label.config(image=self.photo)
        self.counter_label.config(text=f"Image {self.current_index + 1} out of {len(self.images)}")

        # Update radio buttons to match the current image's state
        self.radio_on.config(variable=self.radio_states[self.current_index])
        self.radio_off.config(variable=self.radio_states[self.current_index])

    def update_radio_state(self):
        # This function can be used for any actions triggered by changing the radio button state
        pass

    def toggle_radio_state(self):
        current_var = self.radio_states[self.current_index]
        current_var.set(1 - current_var.get())  # Toggle between 0 and 1

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def show_next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_image()

    def submit_states(self):
        # Collect the current states of the radio buttons
        states = [var.get() for var in self.radio_states.values()]
        self.states_callback(states)
        self.root.destroy()


if __name__ == "__main__":
    direct = "test"
    img_dir = os.path.expanduser(f"../data/{direct}")
    out_dir = os.path.expanduser(f"../results/{direct}")

    image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    images = [Image.open(f).convert("RGB") for f in image_files]
    filenames = [os.path.basename(f) for f in image_files]

    knot_ids = KnotId.group_knots_identify(images)

    df = pd.DataFrame({
        'Filename': filenames,
        'KnotID': knot_ids
    })

    name_first = os.path.splitext(filenames[0])[0]
    name_last = os.path.splitext(filenames[-1])[0]
    csv_pth = os.path.join(out_dir, f'knot_ids_{name_first}-{name_last}.csv')

    df.to_csv(csv_pth, index=False)

    print(f"{len(knot_ids)} Knot IDs saved to {csv_pth}")
    