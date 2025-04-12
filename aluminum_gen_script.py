import os
import random
import bpy
from datetime import datetime
import cv2 as cv
import numpy as np
import sys
import importlib


bl_info = {
    "name": "Data Generator",
    "author": "practice_kubsu_marchuk",
    "version": (1, 0),
    "blender": (3, 6, 2),
    "location": "Output Properties > Data Generator",
    "description": "Генерация синтетических данных для определения дефектов алюминиевых листов",
    "category": "Output",
}


class DATA_GENERATOR_PT_Panel(bpy.types.Panel):
    bl_label = "Data Generator"
    bl_idname = "DATA_GENERATOR_PT_Panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "output"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        output_props = scene.data_generator_props

        layout.prop(output_props, "defect_02")
        layout.prop(output_props, "defect_02_2")
        layout.prop(output_props, "defect_09")
        layout.prop(output_props, "defect_11")
        layout.prop(output_props, "defect_13")
        layout.prop(output_props, "defect_14")
        layout.prop(output_props, "defect_17")
        layout.prop(output_props, "defect_21")
        layout.prop(output_props, "defect_23")
        layout.prop(output_props, "defect_26")
        layout.prop(output_props, "defect_27")
        layout.prop(output_props, "defect_29")
        layout.prop(output_props, "defect_31")
        layout.prop(output_props, "defect_31_2")
        layout.prop(output_props, "defect_32")
        layout.prop(output_props, "defect_34")
        layout.prop(output_props, "defect_36")
        layout.prop(output_props, "defect_39")
        layout.prop(output_props, "defect_41")
        layout.prop(output_props, "defect_57")

        layout.prop(output_props, "dataset_size")
        layout.prop(output_props, "max_tries")
        layout.prop(output_props, "output_path")
        layout.prop(output_props, "use_roughness")
        layout.prop(output_props, "use_normal")
        layout.prop(output_props, "use_displace")
        layout.prop(output_props, "find_contours")
        layout.prop(output_props, "find_bound_box")
        layout.operator("object.data_generator", text="Рендер")


class DATA_GENERATOR_Properties(bpy.types.PropertyGroup):

    defect_02: bpy.props.BoolProperty(
        name="02_поверхностный_пузырь_выпуклый",
        default=False,
    )

    defect_02_2: bpy.props.BoolProperty(
        name="02_поверхностный_пузырь_вогнутый",
        default=False,
    )

    defect_09: bpy.props.BoolProperty(
        name="09_неметаллические_закаты",
        default=False,
    )

    defect_11: bpy.props.BoolProperty(
        name="11_закат_металла",
        default=False,
    )

    defect_13: bpy.props.BoolProperty(
        name="13_коррозия",
        default=False,
    )

    defect_14: bpy.props.BoolProperty(
        name="14_царапины",
        default=False,
    )

    defect_17: bpy.props.BoolProperty(
        name="17_насечка",
        default=False,
    )

    defect_21: bpy.props.BoolProperty(
        name="21_рваная_кромка",
        default=False,
    )

    defect_23: bpy.props.BoolProperty(
        name="23_загар_масла",
        default=False,
    )

    defect_26: bpy.props.BoolProperty(
        name="26_оголение_плакирующего_слоя",
        default=False,
    )

    defect_27: bpy.props.BoolProperty(
        name="27_забоины_вмятины",
        default=False,
    )

    defect_29: bpy.props.BoolProperty(
        name="29_закатанные_царапины",
        default=False,
    )

    defect_31: bpy.props.BoolProperty(
        name="31_отпечатки_на_листах_выпуклые",
        default=False,
    )

    defect_31_2: bpy.props.BoolProperty(
        name="31_отпечатки_на_листах_вогнутые",
        default=False,
    )

    defect_32: bpy.props.BoolProperty(
        name="32_плены",
        default=False,
    )

    # изломы........

    defect_34: bpy.props.BoolProperty(
        name="34_потертость",
        default=False,
    )

    defect_36: bpy.props.BoolProperty(
        name="36_заалюминивание",
        default=False,
    )

    defect_39: bpy.props.BoolProperty(
        name="39_апельсиновая_корка",
        default=False,
    )

    defect_41: bpy.props.BoolProperty(
        name="41_сетка_линий_скольжения",
        default=False,
    )

    defect_57: bpy.props.BoolProperty(
        name="57_следы_СОЖ",
        default=False,
    )

    dataset_size: bpy.props.IntProperty(
        name="Количество изображений",
        default=1,
        min=0,
        max=10000,
        step=1,
    )

    max_tries: bpy.props.IntProperty(
        name="Количество попыток перерендера",
        default=5,
        min=0,
        max=100,
        step=1,
        description="Если в bitmap дефекта менее 100 не чёрных пикселей, "
                    "то сменить Randomness и перерендерить bitmap указанное количество раз"
    )

    output_path: bpy.props.StringProperty(
        name="Вывод",
        subtype="FILE_PATH",
    )

    use_roughness: bpy.props.BoolProperty(
        name="Рендерить шероховатости (Roughness)",
        default=False,
    )

    use_normal: bpy.props.BoolProperty(
        name="Рендерить неровности (Normal)",
        default=False,
    )

    use_displace: bpy.props.BoolProperty(
        name="Рендерить неровности (Displacement)",
        default=False,
    )

    find_contours: bpy.props.BoolProperty(
        name="Найти контуры",
        default=False,
    )

    find_bound_box: bpy.props.BoolProperty(
        name="Найти ограничивающие прямоугольники",
        default=False,
    )


class DATA_GENERATOR_OT_RenderOperator(bpy.types.Operator):
    bl_idname = "object.data_generator"
    bl_label = "Render"

    def execute(self, context):
        """ ОПЕРАТОР """

        props = context.scene.data_generator_props

        path = dir_handler(bpy.path.abspath(props.output_path))
        present_list = get_present_list(props)
        transfer_values_to_bitmaps(present_list, random_only=False)
        inverted_bump_list, bump_list = get_adder_list(present_list)
        unlink_all("aluminum")
        enable_nodes(present_list, "color_adder")
        enable_nodes(inverted_bump_list, "inverted_bump_adder")
        enable_nodes(bump_list, "bump_adder")
        enable_effects(props.use_roughness, props.use_normal, props.use_displace)
        
        nodes_alum = bpy.data.materials["aluminum"].node_tree.nodes
        base_node = nodes_alum.get("metal_base")
        
        render_start = datetime.now()

        for image_idx in range(props.dataset_size):
            base_node.inputs[0].default_value = round(random.uniform(-100, 100), 3)
            transfer_values_to_bitmaps(present_list, random_only=True)
            render_bitmaps(path, present_list, image_idx, props.max_tries, props.find_contours, props.find_bound_box)
            render_aluminum(path, image_idx)

        render_end = datetime.now()
        print(f"Total time: {render_end - render_start}")

        return {"FINISHED"}


def dir_handler(output_path: str) -> str:
    """ Создать папки для вывода отрендеренных изображений и bitmaps """

    now = datetime.now()
    folder = now.strftime("%H-%M-%S %d.%m.%Y")
    path = os.path.join(output_path, folder)
    os.mkdir(path)
    os.mkdir(os.path.join(path, "images"))
    os.mkdir(os.path.join(path, "bitmaps"))
    return path


def enable_effects(use_roughness: bool, use_normal: bool, use_displace: bool):
    """ Подсоединить ноды Roughness и Normal map к шейдеру Principled BSDF """

    links = bpy.data.materials["aluminum"].node_tree.links
    nodes = bpy.data.materials["aluminum"].node_tree.nodes
    principled = nodes["Principled BSDF"]
    if use_roughness:
        links.new(nodes["roughness_bump"].outputs[0], principled.inputs[2])
    else:
        if nodes["roughness_bump"].outputs[0].is_linked:
            links.remove(nodes["roughness_bump"].outputs[0].links[0])

    if use_normal:
        links.new(nodes["roughness_bump"].outputs[1], principled.inputs[5])
    else:
        if nodes["roughness_bump"].outputs[1].is_linked:
            links.remove(nodes["roughness_bump"].outputs[1].links[0])

    if use_displace:
        links.new(nodes["roughness_bump"].outputs[2], nodes["Material Output"].inputs[2])
    else:
        if nodes["roughness_bump"].outputs[2].is_linked:
            links.remove(nodes["roughness_bump"].outputs[2].links[0])


def get_present_list(props) -> list:
    """ Найти список отмеченных дефектов """

    prop_dict = {"02_поверхностный_пузырь_выпуклый": props.defect_02,
                 "02_поверхностный_пузырь_вогнутый": props.defect_02_2,
                 "09_неметаллические_закаты": props.defect_09, "11_закат_металла": props.defect_11,
                 "13_коррозия": props.defect_13,
                 "14_царапины": props.defect_14, "17_насечка": props.defect_17, "21_рваная_кромка": props.defect_21,
                 "23_загар_масла": props.defect_23, "26_оголение_плакирующего_слоя": props.defect_26,
                 "27_забоины": props.defect_27, "29_закатанные_царапины": props.defect_29,
                 "31_отпечатки_на_листах_выпуклые": props.defect_31,
                 "31_отпечатки_на_листах_вогнутые": props.defect_31_2,
                 "32_плены": props.defect_32, "34_потертость": props.defect_34,
                 "36_заалюминивание": props.defect_36, "39_апельсиновая_корка": props.defect_39,
                 "41_сетка_линий_скольжения": props.defect_41,
                 "57_следы_СОЖ": props.defect_57}

    present_list = []
    for defect, prop in prop_dict.items():
        if prop:
            present_list.append(defect)
        else:
            continue

    return present_list


def enable_nodes(nodes_to_link: list, adder: str):
    """ Подключить отмеченные ноды """

    nodes = bpy.data.materials["aluminum"].node_tree.nodes
    links = bpy.data.materials["aluminum"].node_tree.links
    adder_node = bpy.data.materials["aluminum"].node_tree.nodes[adder]

    for name in nodes_to_link:
        if adder == "color_adder":
            output_socket = 0
        else:
            output_socket = 1

        node = nodes.get(name)
        for socket in range(len(adder_node.inputs)):
            if adder_node.inputs[socket].is_linked:
                continue
            else:
                links.new(node.outputs[output_socket], adder_node.inputs[socket])
                break


def transfer_values_to_bitmaps(present_list: list, random_only=True):
    """ Перенести установленные пользователем значения из alum в bitmaps, all - перенос всех параметров """

    nodes_bit = bpy.data.materials["bitmaps"].node_tree.nodes
    nodes_alum = bpy.data.materials["aluminum"].node_tree.nodes
    for name in present_list:
        node, node_to_copy = nodes_bit.get(name), nodes_alum.get(name)
        randomize_node(node_to_copy)
        if not random_only:
            end = 1
        else:
            end = len(node.inputs) - 1
        for i in range(0, len(node.inputs) - end):
            node.inputs[i].default_value = node_to_copy.inputs[i].default_value


def get_adder_list(present_list: list):
    nodes = bpy.data.materials["aluminum"].node_tree.nodes
    inverted_bump_list = []
    for name in present_list:
        node = nodes.get(name)
        if node.inputs[-3].default_value < 0:
            inverted_bump_list.append(name)

    bump_list = [i for i in present_list if i not in inverted_bump_list]
    return inverted_bump_list, bump_list


def randomize_node(node):
    node.inputs[0].default_value = round(random.uniform(-100, 100), 3)


def unlink_all(mat):
    """ Отключить все ноды от входов adder в материале mat """

    links = bpy.data.materials[mat].node_tree.links
    if mat == "bitmaps":
        adder_list = ["adder"]
    else:
        adder_list = ["color_adder", "bump_adder", "inverted_bump_adder"]

    for add in adder_list:
        adder = bpy.data.materials[mat].node_tree.nodes[add]
        for i in range(len(adder.inputs)):
            if adder.inputs[i].is_linked:
                links.remove(adder.inputs[i].links[0])


def render_aluminum(path_to_alum: str, image_indx: int):
    bpy.context.scene.render.filepath = os.path.join(path_to_alum, "images", ("image%d.jpg" % image_indx))
    obj = bpy.data.objects["Plane"]
    obj.active_material = bpy.data.materials["aluminum"]
    bpy.ops.render.render(write_still=True)


def render_bitmaps(path: str, present_list: list, image_indx: int, max_tries: int,
                   find_contours: bool, find_bound_box: bool):
    """ Отрендерить и сохранить bitmaps """

    obj = bpy.data.objects["Plane"]
    obj.active_material = bpy.data.materials["bitmaps"]
    nodes = bpy.data.materials["bitmaps"].node_tree.nodes
    links = bpy.data.materials["bitmaps"].node_tree.links
    adder = bpy.data.materials["bitmaps"].node_tree.nodes["adder"]
    unlink_all("bitmaps")

    for name in present_list:
        node = nodes.get(name)
        link = links.new(node.outputs[0], adder.inputs[0])
        path_to_bitmap = os.path.join(path, "bitmaps", ("bitmap%d_%s.jpg" % (image_indx, name)))
        path_to_csv = os.path.join(path, "images", ("contours%d_%s.csv" % (image_indx, name)))
        bpy.context.scene.render.filepath = path_to_bitmap
        bpy.ops.render.render(write_still=True)
        limit_blank_defects(path_to_bitmap, node, max_tries=max_tries)
        links.remove(link)

        # CONTOURS!
        if find_contours:
            contours.contours_csv(idx=image_indx, image_path=path_to_bitmap, path_to_csv=path_to_csv)
        if find_bound_box:
            contours.bound_box_csv(idx=image_indx, image_path=path_to_bitmap, path_to_csv=path_to_csv)


def limit_blank_defects(path_to_image: str, node, max_tries: int):
    """Предотвратить сохранение изображений, где дефект не виден,
    т.е. перерендерить до max_tries раз """

    tries, whites = 0, 0
    while tries <= max_tries:
        im = cv.imread(path_to_image)
        whites = np.sum(im >= 0)
        if whites >= 100:
            break
        else:
            tries += 1
            randomize_node(node)
            bpy.ops.render.render(write_still=True)


classes = [
    DATA_GENERATOR_PT_Panel,
    DATA_GENERATOR_Properties,
    DATA_GENERATOR_OT_RenderOperator
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.data_generator_props = bpy.props.PointerProperty(type=DATA_GENERATOR_Properties)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.data_generator_props


if __name__ == "__main__":
    import_dir = os.path.dirname(bpy.data.filepath)
    if import_dir not in sys.path:
        sys.path.append(import_dir)

    import contours
    importlib.reload(contours)

    register()
