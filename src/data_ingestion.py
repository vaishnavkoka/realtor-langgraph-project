"""
Real Estate Search Engine - ETL Pipeline
Phase 1: Data Ingestion & Storage

This script implements the complete ETL pipeline:
1. Read Excel file
2. Save canonical row data into PostgreSQL  
3. Process certificates and extract PDF content
4. Prepare data for vector store indexing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Optional
import os
import re

# PDF processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Vector database imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

# Database imports
from database_schema import Property, Certificate, get_session, create_database, get_database_url
from dotenv import load_dotenv
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class PropertyETL:
    """ETL Pipeline for Real Estate Property Data"""
    
    def __init__(self, excel_path: str = None, db_url: str = None):
        self.excel_path = excel_path or os.getenv('PROPERTY_DATA_FILE', './assets/Property_list.xlsx')
        self.db_url = db_url or get_database_url('sqlite')
        self.assets_dir = Path(os.getenv('ASSETS_DIR', './assets'))
        self.certificates_dir = Path(os.getenv('CERTIFICATES_DIR', './assets/certificates'))
        
        # Initialize database
        logger.info(f"Initializing database: {self.db_url}")
        self.engine = create_database(self.db_url)
        self.session = get_session(self.db_url)
        
        # Initialize vector database
        self.vector_db_path = Path(os.getenv('VECTOR_DB_PATH', './vector_db'))
        self.vector_db_path.mkdir(exist_ok=True)
        self.embedding_model = None
        self.faiss_index = None
        self.property_metadata = []
        
        # Statistics
        self.stats = {
            'total_properties': 0,
            'processed_properties': 0,
            'processed_certificates': 0,
            'vectorized_properties': 0,
            'errors': []
        }

    def extract_data(self) -> pd.DataFrame:
        """Step 1: Extract data from Excel file"""
        logger.info(f"Reading Excel file: {self.excel_path}")
        
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"Successfully loaded {len(df)} properties")
            self.stats['total_properties'] = len(df)
            return df
        except Exception as e:
            error_msg = f"Error reading Excel file: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data"""
        logger.info("Cleaning and normalizing data...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Rename the problematic column name
        if 'title / short_description' in df.columns:
            df = df.rename(columns={'title / short_description': 'title'})
        df['city'] = df.apply(
                        lambda row: (
                            match.group(1)
                            if (match := re.search(r'in\s+([A-Za-z]+)', str(row['title']))) 
                            and match.group(1).lower() in str(row['location']).lower()
                            else None
                        ),
                        axis=1
                    )
        df['location'] = df['location'].astype(str).str.replace('\n', ', ', regex=False)
        # Handle missing values
        df['certificates'] = df['certificates'].fillna('')
        df['seller_contact'] = df['seller_contact'].fillna(np.nan)
        df['metadata_tags'] = df['metadata_tags'].fillna('')
        
        # Clean text fields
        text_fields = ['title', 'long_description', 'location', 'metadata_tags']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).str.strip()
        
        # Format phone numbers
        df['seller_contact'] = df['seller_contact'].apply(self.format_phone_number)
        
        # Ensure proper data types
        df['listing_date'] = pd.to_datetime(df['listing_date'])
        
        logger.info("Data cleaning completed")
        return df

    def format_phone_number(self, contact) -> Optional[str]:
        """Format phone numbers - convert 12-digit numbers starting with 91 to +91-XXXXXXXXXX format"""
        if pd.isna(contact):
            return None
            
        # Convert to string and remove any non-digit characters
        contact_str = str(contact).replace('.0', '').strip()
        
        # Remove any existing formatting
        digits_only = re.sub(r'[^\d]', '', contact_str)
        
        # Check if it's a 12-digit number starting with 91 (India country code)
        if len(digits_only) == 12 and digits_only.startswith('91'):
            # Format as +91-XXXXXXXXXX
            return f"+91-{digits_only[2:]}"
        
        # Return as-is for other formats
        return contact_str if contact_str and contact_str != 'nan' else None

    def parse_certificates(self, certificates_str: str) -> List[str]:
        """Parse certificate string into list of filenames"""
        if not certificates_str or pd.isna(certificates_str):
            return []
        
        # Handle multiple certificates separated by |
        certificates = certificates_str.split('|')
        return [cert.strip() for cert in certificates if cert.strip()]

    def extract_pdf_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text content from PDF file"""
        if not PyPDF2:
            logger.warning("PyPDF2 not available, skipping PDF text extraction")
            return None
            
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None
            
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def process_certificates(self, property_id: str, certificates_str: str):
        """Process and store certificate information"""
        certificates = self.parse_certificates(certificates_str)
        
        for cert_filename in certificates:
            cert_path = self.certificates_dir / cert_filename
            
            # Check if certificate already exists
            existing = self.session.query(Certificate).filter_by(
                property_id=property_id, 
                filename=cert_filename
            ).first()
            
            if existing:
                continue
                
            # Extract text content
            extracted_text = None
            file_size = None
            
            if cert_path.exists():
                extracted_text = self.extract_pdf_text(cert_path)
                file_size = cert_path.stat().st_size
            
            # Create certificate record
            certificate = Certificate(
                property_id=property_id,
                filename=cert_filename,
                file_path=str(cert_path),
                extracted_text=extracted_text,
                file_size=file_size,
                is_processed=extracted_text is not None,
                processed_at=datetime.now() if extracted_text else None
            )
            
            self.session.add(certificate)
            self.stats['processed_certificates'] += 1

    def load_properties(self, df: pd.DataFrame):
        """Step 2: Load property data into database"""
        logger.info("Loading properties into database...")
        
        for index, row in df.iterrows():
            try:
                # Check if property already exists
                existing = self.session.query(Property).filter_by(
                    property_id=row['property_id']
                ).first()
                
                if existing:
                    logger.info(f"Property {row['property_id']} already exists, skipping")
                    continue
                
                # Create property record
                property_record = Property(
                    property_id=row['property_id'],
                    num_rooms=int(row['num_rooms']),
                    property_size_sqft=int(row['property_size_sqft']),
                    title=row['title'],
                    long_description=row['long_description'],
                    city = row['city'],
                    location=row['location'],
                    price=int(row['price']),
                    seller_type=row['seller_type'],
                    listing_date=row['listing_date'],
                    certificates=row['certificates'],
                    seller_contact=row['seller_contact'] if pd.notna(row['seller_contact']) else None,
                    metadata_tags=row['metadata_tags']
                )
                
                self.session.add(property_record)
                
                # Process certificates for this property
                if row['certificates']:
                    self.process_certificates(row['property_id'], row['certificates'])
                
                self.stats['processed_properties'] += 1
                
                if (index + 1) % 10 == 0:
                    logger.info(f"Processed {index + 1} properties...")
                    
            except Exception as e:
                error_msg = f"Error processing property {row['property_id']}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
                continue

    def initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, skipping vector database creation")
            return False
            
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def determine_approval_status(self, property_data: Dict) -> str:
        """Determine approval status based on certificates"""
        certificates = property_data.get('certificates', '')
        if not certificates:
            return "Not Approved"
        
        # Count important certificates
        important_certs = ['green-building', 'fire-safety', 'structural-safety','pest-control']
        cert_count = sum(1 for cert in important_certs if cert in certificates.lower())
        
        if cert_count >= 3:
            return "Fully Approved"
        elif cert_count >= 1:
            return "Partially Approved"
        else:
            return "Not Approved"

    def create_property_text_for_embedding(self, property_data: Dict) -> str:
        """Create comprehensive text for embedding including approval status"""
        approval_status = self.determine_approval_status(property_data)
        
        # Create rich text representation
        text_parts = [
            f"Property ID: {property_data.get('property_id', '')}",
            f"Title: {property_data.get('title', '')}",
            f"Description: {property_data.get('long_description', '')}",
            f"Location: {property_data.get('location', '')}",
            f"City: {property_data.get('city', '')}",
            f"Rooms: {property_data.get('num_rooms', '')}",
            f"Size: {property_data.get('property_size_sqft', '')} sqft",
            f"Price: ₹{property_data.get('price', '')}",
            f"Seller Type: {property_data.get('seller_type', '')}",
            f"Contact: {property_data.get('seller_contact', '')}",
            f"Tags: {property_data.get('metadata_tags', '')}",
            f"Approval Status: {approval_status}",
            f"Certificates: {property_data.get('certificates', '')}"
        ]
        
        return " | ".join(filter(None, text_parts))

    def create_vector_database(self):
        """Create FAISS vector database for semantic search"""
        if not faiss or not self.initialize_embedding_model():
            logger.warning("Vector database dependencies not available, skipping")
            return
            
        logger.info("Creating vector database...")
        
        try:
            # Get all properties from database
            properties = self.session.query(Property).all()
            
            if not properties:
                logger.warning("No properties found in database")
                return
            
            # Prepare texts for embedding
            texts = []
            metadata = []
            
            for prop in properties:
                property_data = {
                    'property_id': prop.property_id,
                    'title': prop.title,
                    'long_description': prop.long_description,
                    'location': prop.location,
                    'city': prop.city,
                    'num_rooms': prop.num_rooms,
                    'property_size_sqft': prop.property_size_sqft,
                    'price': prop.price,
                    'seller_type': prop.seller_type,
                    'seller_contact': prop.seller_contact,
                    'metadata_tags': prop.metadata_tags,
                    'certificates': prop.certificates
                }
                
                text_for_embedding = self.create_property_text_for_embedding(property_data)
                texts.append(text_for_embedding)
                
                # Store metadata with approval status
                metadata_entry = property_data.copy()
                metadata_entry['approval_status'] = self.determine_approval_status(property_data)
                metadata.append(metadata_entry)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} properties...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            
            # Save FAISS index and metadata
            faiss_index_path = self.vector_db_path / "property_index.faiss"
            metadata_path = self.vector_db_path / "property_metadata.json"
            
            faiss.write_index(index, str(faiss_index_path))
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.stats['vectorized_properties'] = len(properties)
            logger.info(f"Vector database created successfully with {len(properties)} properties")
            logger.info(f"Index saved to: {faiss_index_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Update database to mark properties as indexed
            for prop in properties:
                prop.is_indexed = True
            
        except Exception as e:
            error_msg = f"Error creating vector database: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

    def run_etl(self):
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Extract
            df = self.extract_data()
            
            # Step 2: Clean
            df = self.clean_data(df)
            
            # Step 3: Load
            self.load_properties(df)
            
            # Step 4: Create vector database
            self.create_vector_database()
            
            # Commit transaction
            self.session.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Print statistics
            self.print_statistics(duration)
            
            logger.info("ETL pipeline completed successfully!")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"ETL pipeline failed: {e}")
            raise
        finally:
            self.session.close()

    def print_statistics(self, duration: float):
        """Print ETL statistics"""
        print("\n" + "=" * 60)
        print("ETL PIPELINE STATISTICS")
        print("=" * 60)
        print(f"Total properties in Excel: {self.stats['total_properties']}")
        print(f"Properties processed: {self.stats['processed_properties']}")
        print(f"Certificates processed: {self.stats['processed_certificates']}")
        print(f"Properties vectorized: {self.stats['vectorized_properties']}")
        print(f"Processing duration: {duration:.2f} seconds")
        print(f"Properties per second: {self.stats['processed_properties']/duration:.2f}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")
        else:
            print("\n No errors encountered!")

def verify_data():
    """Verify the loaded data"""
    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    
    session = get_session(get_database_url('sqlite'))
    
    try:
        # Count properties
        property_count = session.query(Property).count()
        print(f"Properties in database: {property_count}")
        
        # Count certificates
        certificate_count = session.query(Certificate).count()
        print(f"Certificates in database: {certificate_count}")
        
        # Show sample property
        sample = session.query(Property).first()
        if sample:
            print(f"\nSample Property:")
            print(f"  ID: {sample.property_id}")
            print(f"  Title: {sample.title}")
            print(f"  Location: {sample.location}")
            print(f"  Price: ₹{sample.price:,}")
            print(f"  Contact: {sample.seller_contact}")
            print(f"  Description: {sample.long_description[:100]}...")
        
        # Show a property with formatted contact number
        formatted_contact_sample = session.query(Property).filter(
            Property.seller_contact.like('+91-%')
        ).first()
        
        if formatted_contact_sample:
            print(f"\nSample Property with Formatted Contact:")
            print(f"  ID: {formatted_contact_sample.property_id}")
            print(f"  Title: {formatted_contact_sample.title}")
            print(f"  Contact: {formatted_contact_sample.seller_contact}")
            print(f"  Price: ₹{formatted_contact_sample.price:,}")
        
        # Show sample certificate
        cert_sample = session.query(Certificate).first()
        if cert_sample:
            print(f"\nSample Certificate:")
            print(f"  Property ID: {cert_sample.property_id}")
            print(f"  Filename: {cert_sample.filename}")
            print(f"  Processed: {cert_sample.is_processed}")
            if cert_sample.extracted_text:
                print(f"  Text length: {len(cert_sample.extracted_text)} chars")
        
        # Show approval status distribution
        print(f"\n Approval Status Distribution:")
        from pathlib import Path
        vector_db_path = Path('./vector_db')
        metadata_path = vector_db_path / "property_metadata.json"
        
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            approval_counts = {}
            for prop in metadata:
                status = prop.get('approval_status', 'Unknown')
                approval_counts[status] = approval_counts.get(status, 0) + 1
            
            for status, count in approval_counts.items():
                print(f"  {status}: {count} properties")
        
        # Vector database info
        faiss_index_path = vector_db_path / "property_index.faiss"
        if faiss_index_path.exists():
            print(f"\nVector Database:")
            print(f"  FAISS Index: {faiss_index_path}")
            print(f"  Metadata: {metadata_path}")
            print(f"  Status: Ready for semantic search")
        else:
            print(f"\n Vector Database: Not created")
        
    finally:
        session.close()

if __name__ == "__main__":
    # Run ETL pipeline
    etl = PropertyETL()
    etl.run_etl()
    
    # Verify results
    # verify_data() # can verify the data if needed