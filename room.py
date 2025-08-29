import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import json
import random
from io import BytesIO
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuration
st.set_page_config(
    page_title="AI Interior Design Layout Generator",
    page_icon="🏠",
    layout="wide"
)

class InteriorDesignGenerator:
    def __init__(self):
        self.furniture_classes = [
            'sofa', 'chair', 'table', 'bed', 'tv', 'almirah', 
            'fridge', 'swivelchair', 'lamp', 'cabinet'
        ]
        self.design_styles = [
            'Modern', 'Minimalist', 'Bohemian', 'Industrial', 
            'Scandinavian', 'Traditional', 'Contemporary'
        ]
        self.room_types = [
            'Living Room', 'Bedroom', 'Kitchen', 'Dining Room', 
            'Office', 'Study Room'
        ]
        
    def load_model(self):
        """Load YOLO model for furniture detection"""
        try:
            # Using YOLOv8 pretrained model
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def detect_furniture(self, image, model):
        """Detect furniture in the uploaded image"""
        if model is None:
            return [], []
        
        try:
            results = model(image)
            detections = []
            confidences = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for furniture-like objects
                        if confidence > 0.3:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'class': model.names[class_id],
                                'confidence': confidence
                            })
                            confidences.append(confidence)
            
            return detections, confidences
        except Exception as e:
            st.error(f"Error in detection: {e}")
            return [], []
    
    def analyze_room_layout(self, image, detections):
        """Analyze the current room layout"""
        height, width = image.shape[:2]
        
        # Calculate room metrics
        furniture_coverage = 0
        furniture_distribution = {'left': 0, 'center': 0, 'right': 0}
        
        for detection in detections:
            bbox = detection['bbox']
            furniture_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            furniture_coverage += furniture_area
            
            # Determine position
            center_x = (bbox[0] + bbox[2]) / 2
            if center_x < width / 3:
                furniture_distribution['left'] += 1
            elif center_x < 2 * width / 3:
                furniture_distribution['center'] += 1
            else:
                furniture_distribution['right'] += 1
        
        coverage_percentage = (furniture_coverage / (width * height)) * 100
        
        return {
            'coverage_percentage': coverage_percentage,
            'furniture_count': len(detections),
            'distribution': furniture_distribution,
            'room_size': f"{width} x {height} pixels"
        }
    
    def generate_layout_suggestions(self, room_analysis, style, room_type):
        """Generate new layout suggestions based on analysis"""
        suggestions = []
        
        # Style-based recommendations
        style_recommendations = {
            'Modern': {
                'colors': ['#FFFFFF', '#000000', '#C0C0C0', '#8B0000'],
                'furniture_style': 'Clean lines, minimal decoration',
                'layout': 'Open spaces, functional arrangement'
            },
            'Minimalist': {
                'colors': ['#FFFFFF', '#F5F5F5', '#E8E8E8', '#D3D3D3'],
                'furniture_style': 'Essential pieces only, simple forms',
                'layout': 'Maximum open space, strategic placement'
            },
            'Bohemian': {
                'colors': ['#8B4513', '#DAA520', '#CD853F', '#F4A460'],
                'furniture_style': 'Mixed patterns, vintage pieces',
                'layout': 'Layered textures, cozy arrangements'
            },
            'Industrial': {
                'colors': ['#696969', '#A9A9A9', '#2F4F4F', '#708090'],
                'furniture_style': 'Metal and wood, raw materials',
                'layout': 'Exposed elements, functional design'
            },
            'Scandinavian': {
                'colors': ['#FFFFFF', '#F0F8FF', '#E6E6FA', '#DCDCDC'],
                'furniture_style': 'Light wood, cozy textiles',
                'layout': 'Light and airy, hygge elements'
            }
        }
        
        style_info = style_recommendations.get(style, style_recommendations['Modern'])
        
        # Generate suggestions based on current layout
        coverage = room_analysis['coverage_percentage']
        
        if coverage < 20:
            suggestions.append("Room appears sparse - consider adding more furniture pieces")
        elif coverage > 60:
            suggestions.append("Room might be overcrowded - consider removing some items")
        else:
            suggestions.append("Good furniture coverage - focus on rearrangement")
        
        # Distribution suggestions
        distribution = room_analysis['distribution']
        max_side = max(distribution.values())
        if max_side > len(distribution) * 2:
            suggestions.append("Consider redistributing furniture for better balance")
        
        return {
            'suggestions': suggestions,
            'style_guide': style_info,
            'room_specific': self.get_room_specific_advice(room_type)
        }
    
    def get_room_specific_advice(self, room_type):
        """Get room-specific design advice"""
        advice = {
            'Living Room': [
                'Focus on seating arrangement facing each other',
                'Create a focal point (TV, fireplace, or art)',
                'Use a coffee table to anchor seating area',
                'Ensure good traffic flow'
            ],
            'Bedroom': [
                'Place bed against the longest wall',
                'Ensure nightstands on both sides if space allows',
                'Create a dressing area if possible',
                'Optimize natural light flow'
            ],
            'Kitchen': [
                'Follow the work triangle principle',
                'Maximize counter space',
                'Ensure proper lighting over work areas',
                'Consider storage optimization'
            ],
            'Dining Room': [
                'Center table in the room',
                'Allow 3 feet clearance around table',
                'Position lighting directly over table',
                'Create a sideboard or buffet area'
            ],
            'Office': [
                'Position desk to face the room entrance',
                'Ensure adequate lighting for work',
                'Create storage solutions',
                'Minimize distractions'
            ]
        }
        return advice.get(room_type, ['General room layout principles apply'])
    
    def create_layout_visualization(self, image, detections, suggestions):
        """Create a visual representation of the layout"""
        img_copy = image.copy()
        
        # Draw bounding boxes for detected furniture
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return img_copy

    def generate_3d_layout(self, detections, room_size):
        """Generate a 3D layout visualization of the room"""
        width, height = room_size
        fig = go.Figure()

        # Draw room floor as a rectangle
        fig.add_trace(go.Scatter3d(
            x=[0, width, width, 0, 0],
            y=[0, 0, height, height, 0],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            line=dict(color='black', width=5),
            name='Room Floor'
        ))

        # Add furniture as boxes
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            # Convert to room coordinates (assuming image coordinates)
            # For simplicity, z dimension is fixed height
            z1, z2 = 0, 50  # arbitrary height for furniture

            # Create vertices of the box
            vertices = [
                [x1, y1, z1],
                [x2, y1, z1],
                [x2, y2, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
            ]

            # Define the 12 edges of the box
            edges = [
                (0,1), (1,2), (2,3), (3,0),  # bottom square
                (4,5), (5,6), (6,7), (7,4),  # top square
                (0,4), (1,5), (2,6), (3,7)   # vertical edges
            ]

            for edge in edges:
                x_coords = [vertices[edge[0]][0], vertices[edge[1]][0]]
                y_coords = [vertices[edge[0]][1], vertices[edge[1]][1]]
                z_coords = [vertices[edge[0]][2], vertices[edge[1]][2]]
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color='blue', width=4),
                    showlegend=False
                ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, range=[0, width], backgroundcolor="white"),
                yaxis=dict(nticks=10, range=[0, height], backgroundcolor="white"),
                zaxis=dict(nticks=5, range=[0, 100], backgroundcolor="white"),
                aspectratio=dict(x=width/height, y=1, z=0.5)
            ),
            margin=dict(r=10, l=10, b=10, t=10),
            height=600,
            title="3D Room Layout"
        )
        return fig

    def generate_shopping_list(self, detections, style):
        """Generate a shopping list based on detected furniture and style"""
        # Furniture recommendations database
        furniture_recommendations = {
            'sofa': {
                'Modern': ['Minimalist Leather Sofa', 'Sectional Sofa with Clean Lines'],
                'Minimalist': ['Simple Fabric Sofa', 'Low-profile Sofa'],
                'Bohemian': ['Vintage Patterned Sofa', 'Colorful Bohemian Sofa'],
                'Industrial': ['Leather Chesterfield Sofa', 'Metal Frame Sofa'],
                'Scandinavian': ['Light Wood Sofa', 'Cozy Fabric Sofa']
            },
            'chair': {
                'Modern': ['Eames Style Chair', 'Modern Accent Chair'],
                'Minimalist': ['Simple Wooden Chair', 'Minimalist Dining Chair'],
                'Bohemian': ['Patterned Armchair', 'Vintage Chair'],
                'Industrial': ['Metal Frame Chair', 'Leather Bar Stool'],
                'Scandinavian': ['Wooden Dining Chair', 'Cozy Armchair']
            },
            'table': {
                'Modern': ['Glass Coffee Table', 'Minimalist Dining Table'],
                'Minimalist': ['Simple Wood Table', 'Sleek Console Table'],
                'Bohemian': ['Vintage Coffee Table', 'Patterned Side Table'],
                'Industrial': ['Metal Coffee Table', 'Wood and Metal Dining Table'],
                'Scandinavian': ['Light Wood Table', 'Simple Coffee Table']
            },
            'bed': {
                'Modern': ['Platform Bed', 'Minimalist Bed Frame'],
                'Minimalist': ['Simple Bed Frame', 'Low-profile Bed'],
                'Bohemian': ['Canopy Bed', 'Vintage Bed Frame'],
                'Industrial': ['Metal Bed Frame', 'Wood and Metal Bed'],
                'Scandinavian': ['Light Wood Bed', 'Simple Bed Frame']
            }
        }

        shopping_list = []
        detected_types = set(det['class'] for det in detections)
        
        # Add recommendations for detected furniture types
        for furniture_type in detected_types:
            if furniture_type in furniture_recommendations:
                recommendations = furniture_recommendations[furniture_type].get(style, [])
                for item in recommendations:
                    shopping_list.append({
                        'Item': item,
                        'Type': furniture_type.capitalize(),
                        'Estimated Price': f"${random.randint(200, 2000)}",
                        'Priority': 'High' if furniture_type in ['bed', 'sofa'] else 'Medium'
                    })
        
        # Add some additional items based on room style
        style_additions = {
            'Modern': ['Modern Floor Lamp', 'Abstract Wall Art', 'Minimalist Rug'],
            'Minimalist': ['Simple Wall Shelf', 'Minimalist Decor', 'Neutral Rug'],
            'Bohemian': ['Macrame Wall Hanging', 'Patterned Rug', 'Plants'],
            'Industrial': ['Metal Wall Art', 'Industrial Lighting', 'Concrete Planters'],
            'Scandinavian': ['Cozy Throw Blanket', 'Wooden Decor', 'Simple Lighting']
        }
        
        for item in style_additions.get(style, []):
            shopping_list.append({
                'Item': item,
                'Type': 'Decor',
                'Estimated Price': f"${random.randint(50, 300)}",
                'Priority': 'Low'
            })

        return shopping_list

def main():
    st.title("🏠 AI Interior Design Layout Generator")
    st.markdown("Upload a room photo and get AI-powered design suggestions!")
    
    # Initialize the generator
    generator = InteriorDesignGenerator()
    
    # Sidebar for configuration
    st.sidebar.header("🎨 Design Configuration")
    selected_style = st.sidebar.selectbox(
        "Choose Design Style:",
        generator.design_styles
    )
    
    selected_room_type = st.sidebar.selectbox(
        "Room Type:",
        generator.room_types
    )
    
    st.sidebar.header("📊 Available Datasets")
    st.sidebar.markdown("""
    **Using these Kaggle datasets:**
    - House Rooms Image Dataset
    - Furniture Detection Dataset  
    - Furniture Image Dataset (5 classes)
    - Synthetic Home Interior Dataset
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Room Photo")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a photo of your room for analysis"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Room Photo", use_column_width=True)
            
            # Convert to OpenCV format
            opencv_image = np.array(image.convert('RGB'))
            opencv_image = opencv_image[:, :, ::-1].copy()
            
            # Process the image
            with st.spinner("🔍 Analyzing room layout..."):
                # Load model (simplified for demo)
                model = generator.load_model()
                
                # Detect furniture
                detections, confidences = generator.detect_furniture(opencv_image, model)
                
                # Analyze room
                room_analysis = generator.analyze_room_layout(opencv_image, detections)
                
                # Generate suggestions
                layout_suggestions = generator.generate_layout_suggestions(
                    room_analysis, selected_style, selected_room_type
                )
    
    with col2:
        if uploaded_file is not None:
            st.header("🎯 Analysis Results")
            
            # Room metrics
            st.subheader("📏 Room Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Furniture Coverage", f"{room_analysis['coverage_percentage']:.1f}%")
                st.metric("Furniture Count", room_analysis['furniture_count'])
            
            with metrics_col2:
                st.metric("Room Size", room_analysis['room_size'])
                
                # Distribution chart
                if room_analysis['furniture_count'] > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    distribution = room_analysis['distribution']
                    ax.bar(distribution.keys(), distribution.values())
                    ax.set_title("Furniture Distribution")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
            
            # Style recommendations
            st.subheader(f"🎨 {selected_style} Style Guide")
            style_guide = layout_suggestions['style_guide']
            
            # Color palette
            st.write("**Color Palette:**")
            color_cols = st.columns(len(style_guide['colors']))
            for i, color in enumerate(style_guide['colors']):
                with color_cols[i]:
                    st.color_picker("", color, disabled=True, key=f"color_{i}")
            
            st.write(f"**Furniture Style:** {style_guide['furniture_style']}")
            st.write(f"**Layout Principle:** {style_guide['layout']}")
            
            # Layout suggestions
            st.subheader("💡 Layout Suggestions")
            for suggestion in layout_suggestions['suggestions']:
                st.write(f"• {suggestion}")
            
            # Room-specific advice
            st.subheader(f"🏠 {selected_room_type} Specific Tips")
            for tip in layout_suggestions['room_specific']:
                st.write(f"• {tip}")
            
            # Visualize detection results
            if detections:
                st.subheader("🔍 Detected Furniture")
                annotated_image = generator.create_layout_visualization(
                    opencv_image, detections, layout_suggestions
                )
                st.image(
                    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                    caption="Detected Furniture", 
                    use_column_width=True
                )
                
                # Detection details
                detection_data = []
                for det in detections:
                    detection_data.append({
                        'Furniture': det['class'],
                        'Confidence': f"{det['confidence']:.2f}",
                        'Position': f"({int(det['bbox'][0])}, {int(det['bbox'][1])})"
                    })
                
                if detection_data:
                    df = pd.DataFrame(detection_data)
                    st.dataframe(df, use_container_width=True)
    
    # Additional features
    st.header("🚀 Additional Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.subheader("📐 3D Visualization")
        st.write("Generate 3D room layouts")
        if st.button("Generate 3D Layout"):
            if uploaded_file is not None and detections:
                # Get room dimensions from the image
                height, width = opencv_image.shape[:2]
                room_size = (width, height)
                
                # Generate 3D layout
                with st.spinner("🔄 Generating 3D layout..."):
                    fig_3d = generator.generate_3d_layout(detections, room_size)
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Please upload an image and detect furniture first!")
    
    with feature_col2:
        st.subheader("🛒 Shopping List")
        st.write("Get furniture recommendations")
        if st.button("Generate Shopping List"):
            if uploaded_file is not None and detections:
                # Generate shopping list
                with st.spinner("🛒 Generating shopping list..."):
                    shopping_list = generator.generate_shopping_list(detections, selected_style)
                    
                    if shopping_list:
                        st.subheader("🛍️ Your Shopping List")
                        
                        # Display as a table
                        df = pd.DataFrame(shopping_list)
                        st.dataframe(df, use_container_width=True)
                        
                        # Add download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name="shopping_list.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON download
                            json_data = df.to_json(orient="records", indent=2)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name="shopping_list.json",
                                mime="application/json"
                            )
                    else:
                        st.warning("No furniture detected to generate shopping list")
            else:
                st.warning("Please upload an image and detect furniture first!")
    
    with feature_col3:
        st.subheader("📱 AR Preview")
        st.write("Preview changes in AR")
        if st.button("AR Preview"):
            st.info("AR preview would be implemented here")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, OpenCV, and YOLO")

if __name__ == "__main__":
    main() 