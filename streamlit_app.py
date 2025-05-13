import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import traceback

class NumerologyGridMaker:
    def __init__(self):
        # Constants
        self.YEAR_DAYS = 365.25
        self.MAX_AGE = 135
        
        # Day coordinates mapping (Sun=0, Mon=1, etc. in Python's datetime)
        self.day_coordinates = [1, 2, 9, 5, 3, 6, 8]  # Sun, Mon, Tue, Wed, Thu, Fri, Sat
        self.day_num_to_name = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        
        # Grid definition (Nadi style)
        self.grid_style = 'nadi'
        self.grid_definition = {'nadi': ['3', '1', '9', '6', '7', '5', '2', '8', '4']}
        self.dasha_sequence = {'nadi': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        
        # State variables
        self.dashas = {}
        self.curr_index = 0
        self.curr_antra = 0
        self.start_dasha = -1
        self.start_antra = -1
    
    def format_date(self, d):
        """Format date as DD/MM/YYYY"""
        return f"{d.day}/{d.month}/{d.year}"
    
    def reduce_to_single(self, num):
        """Reduce a number to a single digit by summing its digits"""
        if num == 0:
            return 0
            
        # Make sure we're working with a positive integer
        num = abs(int(num))
        
        result = 0
        while num > 0:
            result += num % 10
            num = num // 10
            
        # Recursively reduce until we get a single digit
        if result > 9:
            return self.reduce_to_single(result)
        return result
    
    def create_dashas(self, birth_date):
        """Create Dasha periods based on birth date"""
        try:
            d = birth_date.day
            m = birth_date.month
            y = birth_date.year
            
            # Initialize the dasha dictionary
            td = {}
            self.dashas['mahaDasha'] = td
            td['index'] = 0
            td['dob'] = birth_date
            
            # Calculate basic and destiny numbers
            td['basic'] = self.reduce_to_single(d)
            destiny_num = int(f"{d}{m}{y}")
            td['destiny'] = self.reduce_to_single(destiny_num)
            
            # Calculate grid numbers
            g = 0 if (d < 10 or d % 10 == 0) else td['basic']
            td['basicGrid'] = f"{d}{m}{y % 100}{td['destiny']}{g}"
            
            # Get Dasha details
            td['details'] = self.get_dasha_for(td['basic'], 45 * self.YEAR_DAYS, td['dob'])
            
            basic_info = f"Basic: {td['basic']}, Destiny: {td['destiny']}"
            
            return td['details'], basic_info
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error in create_dashas: {e}")
            print(f"Error in create_dashas: {e}\n{error_details}")
            raise
    
    def get_dasha_for(self, num, total_days, birth_date):
        """Calculate Dasha periods based on birth number"""
        try:
            # Get Dasha sequence
            yseq = self.dasha_sequence[self.grid_style]
            
            # Make sure num is in the sequence
            if num not in yseq:
                num = 1  # Default to first element if not found
                
            i = yseq.index(num)
            ed = birth_date
            
            # Create sequence starting from the birth number
            desq = yseq[i:] + yseq[:i]
            desq = desq * 3  # Repeat to cover max age
            
            # Current date for determining current period
            curr_date = date.today()
            
            grid_years = []
            g = 0
            
            # Get the basic grid from mahaDasha
            basic_grid = self.dashas['mahaDasha']['basicGrid']
            
            self.start_dasha = -1
            self.start_antra = -1
            
            for n, dasha_val in enumerate(desq):
                if g > self.MAX_AGE:
                    break
                    
                det = {}
                
                # Generate numbers for the grid
                f = dasha_val
                det['num'] = list(basic_grid.replace('0', ''))
                
                # Calculate start and end dates for this period
                sd = ed
                
                # Safely calculate end date
                try:
                    ed = date(sd.year + dasha_val, sd.month, sd.day)
                except ValueError:
                    # Handle leap year issue for Feb 29
                    if sd.month == 2 and sd.day == 29:
                        ed = date(sd.year + dasha_val, 3, 1)
                    else:
                        raise
                
                det['grids'] = []
                det['grid'] = {}
                
                # Populate the grid
                for i in range(len(det['num'])):
                    digit = det['num'][i]
                    if det['grid'].get(digit) is None:
                        det['grid'][digit] = ''
                    det['grid'][digit] += digit
                
                # Add dasha number to grid
                str_f = str(f)
                if det['grid'].get(str_f) is None:
                    det['grid'][str_f] = ''
                det['grid'][str_f] += f'<{f}>'  # Marking with <> for dasha styling
                
                det['start'] = sd
                det['end'] = ed - timedelta(days=1)
                det['duration'] = dasha_val
                det['daycount'] = total_days * dasha_val
                det['age'] = g
                
                # Check if this is the current period
                if sd <= curr_date <= det['end'] and self.start_dasha == -1:
                    self.start_dasha = n
                
                # Calculate Antra-Dashas (yearly sub-periods)
                sconst = self.reduce_to_single(birth_date.day + birth_date.month + 1)
                sde = sd
                
                for i in range(dasha_val):
                    buk = {}
                    buk['dshcoord'] = dasha_val
                    buk['start'] = sde
                    
                    # Calculate coordinate from day of week and year
                    weekday = sde.weekday()
                    day_of_week = (weekday + 1) % 7  # Adjust to 0=Sunday
                    
                    # Safety checks for array access
                    if day_of_week < 0 or day_of_week >= len(self.day_coordinates):
                        day_of_week = 0
                        
                    day_coord = self.day_coordinates[day_of_week]
                    day_name = self.day_num_to_name[day_of_week]
                    
                    buk['num'] = f"{sconst}{day_coord}{sde.year % 100}"
                    buk['day'] = day_name
                    buk['coord'] = self.reduce_to_single(int(buk['num']))
                    
                    # Calculate end date safely
                    try:
                        sde_next = date(sde.year + 1, sde.month, sde.day)
                    except ValueError:
                        # Handle leap year issue for Feb 29
                        if sde.month == 2 and sde.day == 29:
                            sde_next = date(sde.year + 1, 3, 1)
                        else:
                            raise
                            
                    buk['end'] = sde_next - timedelta(days=1)
                    
                    # Copy dasha grid and add antra-dasha number
                    buk['grid'] = det['grid'].copy()
                    coord_str = str(buk['coord'])
                    if buk['grid'].get(coord_str) is None:
                        buk['grid'][coord_str] = ''
                    buk['grid'][coord_str] += f'[{buk["coord"]}]'  # Marking with [] for antra styling
                    
                    det['grids'].append(buk)
                    
                    # Check if this is the current antra period
                    if buk['start'] <= curr_date <= buk['end'] and self.start_antra == -1:
                        self.start_antra = i
                    
                    # Move to next year
                    sde = sde_next
                
                grid_years.append(det)
                g += dasha_val
            
            # Use default values if we couldn't determine the current period
            if self.start_dasha == -1:
                self.start_dasha = 0
            if self.start_antra == -1:
                self.start_antra = 0
                
            # Make sure we have at least one period
            if not grid_years:
                # Create a default period if none were calculated
                default_det = {
                    'start': birth_date,
                    'end': birth_date + timedelta(days=365),
                    'duration': 1,
                    'daycount': 365,
                    'age': 0,
                    'grid': {},
                    'grids': [{
                        'dshcoord': 1,
                        'start': birth_date,
                        'end': birth_date + timedelta(days=365),
                        'day': 'Sun',
                        'coord': 1,
                        'grid': {}
                    }]
                }
                grid_years.append(default_det)
                
            return grid_years
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error in get_dasha_for: {e}")
            print(f"Error in get_dasha_for: {e}\n{error_details}")
            raise
    
    def create_grid_visualization(self, dasha_index, antra_index):
        """Create a visual representation of the numerology grid using Plotly"""
        try:
            if not self.dashas.get('mahaDasha', {}).get('details'):
                return None
            
            grid_years = self.dashas['mahaDasha']['details']
            if not grid_years:
                return None
                
            # Convert indices to int if they're strings
            if isinstance(dasha_index, str):
                try:
                    # Extract the number before the colon
                    dasha_index = int(dasha_index.split(':')[0])
                except (ValueError, IndexError):
                    dasha_index = 0
                    
            if isinstance(antra_index, str):
                try:
                    antra_index = int(antra_index)
                except ValueError:
                    antra_index = 0
            
            # Validate indices
            dasha_index = max(min(dasha_index, len(grid_years)-1), 0)
            
            if dasha_index >= len(grid_years):
                return None
                
            # Get the current grid
            dasha = grid_years[dasha_index]
            
            if not dasha.get('grids'):
                return None
                
            antra_index = max(min(antra_index, len(dasha['grids'])-1), 0)
            
            if antra_index >= len(dasha['grids']):
                return None
                
            antra = dasha['grids'][antra_index]
            gm = antra.get('grid', {})
            seq = self.grid_definition[self.grid_style]
            
            # Create a 3x3 grid for visualization
            grid = np.zeros((3, 3), dtype=object)
            grid_colors = np.full((3, 3), 'white', dtype=object)
            
            # Fill the grid with values
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(seq):
                        cell_value = gm.get(seq[idx], '')
                        
                        # Create formatted text for each cell
                        formatted_text = cell_value
                        
                        # Save cell value to grid
                        grid[i, j] = formatted_text
                        
                        # Set color based on content
                        if '<' in str(cell_value) and '[' in str(cell_value):
                            grid_colors[i, j] = '#e6f2ff'  # Light blue if contains both
                        elif '<' in str(cell_value):
                            grid_colors[i, j] = '#e6f9ff'  # Very light blue if contains dasha
                        elif '[' in str(cell_value):
                            grid_colors[i, j] = '#ffebeb'  # Very light red if contains antra
            
            # Create figure
            fig = go.Figure()
            
            # Add squares for each cell
            for i in range(3):
                for j in range(3):
                    # Format cell content for display
                    cell_content = str(grid[i, j])
                    cell_content = cell_content.replace('<', '<b style="color:blue;">')
                    cell_content = cell_content.replace('>', '</b>')
                    cell_content = cell_content.replace('[', '<b style="color:red;">')
                    cell_content = cell_content.replace(']', '</b>')
                    
                    fig.add_trace(go.Scatter(
                        x=[j, j+1, j+1, j, j],
                        y=[3-i, 3-i, 3-(i+1), 3-(i+1), 3-i],
                        fill="toself",
                        fillcolor=grid_colors[i, j],
                        line=dict(color='black', width=1),
                        mode='lines',
                        showlegend=False,
                        hoverinfo='text',
                        text=f"Cell {seq[i*3 + j] if i*3 + j < len(seq) else 'N/A'}"
                    ))
                    
                    # Add text
                    fig.add_annotation(
                        x=j+0.5,
                        y=3-(i+0.5),
                        text=cell_content,
                        showarrow=False,
                        font=dict(size=14)
                    )
            
            # Set figure layout
            fig.update_layout(
                title=f"Numerology Grid: Dasha {antra.get('dshcoord', 'N/A')}, Antra {antra.get('coord', 'N/A')}",
                width=400,
                height=400,
                showlegend=False,
                plot_bgcolor='white',
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 3]),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 3])
            )
            
            return fig
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error in create_grid_visualization: {e}")
            print(f"Error in create_grid_visualization: {e}\n{error_details}")
            return None
    
    def get_dasha_list(self):
        """Generate list of dasha periods for dropdown"""
        try:
            if not self.dashas.get('mahaDasha', {}).get('details'):
                return []
                
            grid_years = self.dashas['mahaDasha']['details']
            dasha_list = []
            
            for i, period in enumerate(grid_years):
                dasha_list.append(f"{i}: From: {self.format_date(period.get('start', date.today()))} - {self.format_date(period.get('end', date.today()))} | Age: {period.get('age', 0)} - {period.get('duration', 0) + period.get('age', 0)} yrs")
            
            return dasha_list
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error in get_dasha_list: {e}")
            print(f"Error in get_dasha_list: {e}\n{error_details}")
            return ["Error generating dasha list"]
    
    def display_grid(self, dasha_index, antra_index):
        """Generate HTML for displaying the grid in Streamlit"""
        try:
            if not self.dashas.get('mahaDasha', {}).get('details'):
                st.warning("Please enter a birth date")
                return
            
            grid_years = self.dashas['mahaDasha']['details']
            if not grid_years:
                st.warning("No grid data available")
                return
            
            # Use current or provided indices
            # Convert to int if it's a string (from dropdown)
            if isinstance(dasha_index, str):
                try:
                    # Extract the number before the colon
                    dasha_index = int(dasha_index.split(':')[0])
                except (ValueError, IndexError):
                    dasha_index = 0
            
            # Make sure indices are valid
            dasha_index = max(min(dasha_index, len(grid_years)-1), 0)
            
            if dasha_index >= len(grid_years):
                st.warning("Dasha index out of range")
                return
                
            dasha = grid_years[dasha_index]
            
            if not dasha.get('grids'):
                st.warning("No antra periods available")
                return
                
            antra_index = max(min(antra_index, len(dasha['grids'])-1), 0)
            
            if antra_index >= len(dasha['grids']):
                st.warning("Antra index out of range")
                return
                
            antra = dasha['grids'][antra_index]
            
            # Display period info
            st.subheader(f"Period Information")
            st.write(f"Dasha: {antra.get('dshcoord', 'N/A')} | Antra: {antra.get('coord', 'N/A')}")
            st.write(f"Period: {self.format_date(antra.get('start', date.today()))} to {self.format_date(antra.get('end', date.today()))}")
            
            # Get grid data
            seq = self.grid_definition[self.grid_style]
            gm = antra.get('grid', {})
            
            # Display the grid using HTML and CSS
            grid_html = """
            <style>
                .grid-container {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    grid-gap: 2px;
                    margin: 20px 0;
                }
                .grid-cell {
                    border: 1px solid black;
                    padding: 15px;
                    text-align: center;
                    height: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background-color: white;
                }
                .dasha {
                    color: blue;
                    font-weight: bold;
                }
                .antra {
                    color: red;
                    font-weight: bold;
                }
            </style>
            <div class="grid-container">
            """
            
            for cell_num in seq:
                cell_content = gm.get(cell_num, '')
                # Format dasha and antra markers
                cell_content = str(cell_content)
                cell_content = cell_content.replace('<', '<span class="dasha">').replace('>', '</span>')
                cell_content = cell_content.replace('[', '<span class="antra">').replace(']', '</span>')
                
                grid_html += f'<div class="grid-cell">{cell_content}</div>'
            
            grid_html += "</div>"
            
            st.markdown(grid_html, unsafe_allow_html=True)
            
            # Display plotly visualization as well
            fig = self.create_grid_visualization(dasha_index, antra_index)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display antra list
            self.display_antra_list(dasha_index)
            
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error displaying grid: {e}")
            print(f"Error in display_grid: {e}\n{error_details}")
    
    def display_antra_list(self, dasha_index):
        """Display antra-dasha list for the selected dasha period"""
        try:
            if not self.dashas.get('mahaDasha', {}).get('details'):
                return
                
            grid_years = self.dashas['mahaDasha']['details']
            if not grid_years:
                return
            
            # Convert to int if it's a string (from dropdown)
            if isinstance(dasha_index, str):
                try:
                    # Extract the number before the colon
                    dasha_index = int(dasha_index.split(':')[0])
                except (ValueError, IndexError):
                    dasha_index = 0
                    
            # Ensure dasha_index is valid
            dasha_index = max(min(dasha_index, len(grid_years)-1), 0)
            
            if dasha_index >= len(grid_years):
                return
                
            antra_list = grid_years[dasha_index].get('grids', [])
            
            st.subheader("Antra-Dasha Periods")
            
            # Create a dataframe for the antra list
            antra_data = []
            curr_date = date.today()
            
            for i, antra in enumerate(antra_list):
                is_current = False
                if 'start' in antra and 'end' in antra:
                    is_current = antra['start'] <= curr_date <= antra['end']
                
                antra_data.append({
                    'Index': i,
                    'Period': f"{self.format_date(antra.get('start', date.today()))} - {self.format_date(antra.get('end', date.today()))}",
                    'Day': antra.get('day', 'N/A'),
                    'Antra Number': antra.get('coord', 'N/A'),
                    'Current': '✓' if is_current else ''
                })
            
            if antra_data:
                df = pd.DataFrame(antra_data)
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.write("No Antra-Dasha periods available")
                
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error displaying antra list: {e}")
            print(f"Error in display_antra_list: {e}\n{error_details}")

# Initialize the app
st.set_page_config(page_title="Numerology Grid Maker", layout="wide")

# Add app title
st.title("Numerology Grid Maker")
st.write("Enter your birth date to generate your numerology grid")

# Create sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Date input
    birth_date = st.date_input(
        "Birth Date",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        value=date(1990, 1, 1),
        help="Select your birth date"
    )
    
    # Calculate button
    calculate_button = st.button("Calculate Grid", type="primary")
    
    # Initialize session state for storing calculator
    if 'calculator' not in st.session_state:
        st.session_state.calculator = None
        st.session_state.basic_info = None
        st.session_state.dasha_list = []
    
    # Dasha and Antra selection (only shown after calculation)
    if st.session_state.calculator is not None:
        st.divider()
        st.subheader("Period Selection")
        
        dasha_index = st.selectbox(
            "Dasha Period",
            options=st.session_state.dasha_list,
            index=0 if st.session_state.dasha_list else None
        )
        
        antra_index = st.number_input(
            "Antra Index",
            min_value=0,
            value=0,
            step=1,
            help="Select antra-dasha period index"
        )
        
        update_button = st.button("Update View")
    
    # Add troubleshooting section
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### Common Issues:
        
        1. **No Grid Appears**: Check that your birth date is valid
        2. **Calculation Errors**: Try a different birth date
        3. **Mobile Display Issues**: Try landscape orientation
        
        If you encounter persistent issues, check the console output for detailed error messages.
        """)

# Main calculation logic
try:
    # When calculate button is pressed
    if calculate_button:
        with st.spinner("Calculating numerology grid..."):
            # Initialize calculator
            calculator = NumerologyGridMaker()
            
            # Calculate grid
            grid_years, basic_info = calculator.create_dashas(birth_date)
            
            # Store in session state
            st.session_state.calculator = calculator
            st.session_state.basic_info = basic_info
            st.session_state.dasha_list = calculator.get_dasha_list()
            
            # Show success message
            st.sidebar.success("Grid calculated successfully!")
    
    # Display results
    if st.session_state.calculator is not None:
        # Display basic info
        st.header("Numerology Details")
        st.info(st.session_state.basic_info)
        
        # Check if update button exists and is pressed or if we just calculated
        update_grid = False
        if 'update_button' in locals() and update_button:
            update_grid = True
        elif calculate_button:
            update_grid = True
            dasha_index = 0
            antra_index = 0
        
        if update_grid:
            # Display grid for selected period
            st.session_state.calculator.display_grid(dasha_index, antra_index)
        elif calculate_button:
            # Display initial grid
            st.session_state.calculator.display_grid(0, 0)
        else:
            # Display the last viewed grid
            st.session_state.calculator.display_grid(0, 0)
    
except Exception as e:
    error_details = traceback.format_exc()
    st.error(f"An error occurred: {str(e)}")
    print(f"Error in main app flow: {e}\n{error_details}")

# Footer
st.divider()
st.write("Numerology Grid Maker | Created with Streamlit")
