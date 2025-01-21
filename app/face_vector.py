import psycopg2
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import execute_values

class FaceEmbeddingDB:
    def __init__(self, db_params: Dict[str, str]):
        """
        Initialize database connection with connection parameters.
        
        Args:
            db_params: Dictionary containing database connection parameters
                      (dbname, user, password, host, port)
        """
        self.db_params = db_params
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print("Successfully connected to the database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        create_tables_query = """
        -- Create face_embeddings table
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            embedding vector(128) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.conn.cursor() as cur:
            cur.execute(create_tables_query)
            self.conn.commit()

    def store_embedding(self, name: str, embedding: np.ndarray) -> bool:
        """
        Store a face embedding for an employee.
        
        Args:
            name: Employee name
            embedding: Face embedding numpy array
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings (name, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE 
                    SET embedding = EXCLUDED.embedding
                """, (name, embedding.tolist()))
                
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            self.conn.rollback()
            return False

    def store_multiple_embeddings(self, embeddings_data: List[Dict[str, any]]) -> bool:
        """
        Store multiple face embeddings in batch.
        
        Args:
            embeddings_data: List of dictionaries containing name and embedding
                           [{'name': 'John', 'embedding': np_array}, ...]
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.conn.cursor() as cur:
                embeddings_values = [
                    (data['name'], data['embedding'].tolist())
                    for data in embeddings_data
                ]
                
                execute_values(cur, """
                    INSERT INTO face_embeddings (name, embedding)
                    VALUES %s
                    ON CONFLICT (name) DO UPDATE 
                    SET embedding = EXCLUDED.embedding
                """, embeddings_values)
                
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error storing multiple embeddings: {e}")
            self.conn.rollback()
            return False
        
    def retrieve_all_data(self) -> List[Dict[str, any]]:
        """
        Retrieve all data from face_embeddings table.
        
        Returns:
            List[Dict[str, any]]: List containing all data from the table
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM face_embeddings")
                face_embeddings = cur.fetchall()
                face_embeddings_data = [
                    {"id": row[0], "name": row[1], "embedding": row[2], "created_at": row[3]}
                    for row in face_embeddings
                ]
                
                return face_embeddings_data
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return []
    def delete_tables(self):
        """Delete face_embeddings table."""
        delete_tables_query = "DROP TABLE IF EXISTS face_embeddings;"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_tables_query)
                self.conn.commit()
                print("Successfully deleted the table")
        except Exception as e:
            print(f"Error deleting table: {e}")
            self.conn.rollback()
    
    def embedding_exists(self, name: str) -> bool:
        """
        Check if an embedding for the given employee name exists in the database.
        
        Args:
            name: Employee name
        
        Returns:
            bool: True if the embedding exists, False otherwise
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM face_embeddings WHERE name = %s", (name,))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if embedding exists: {e}")
            return False
            
    def vector_search(self, encoding: np.ndarray) -> List[Dict[str, any]]:
        """
        Search for the top 5 closest face embeddings to the given encoding.
        
        Args:
            encoding: Face encoding numpy array
        
        Returns:
            List[Dict[str, any]]: List of top 5 closest matching face embedding data
        """
        try:
            with self.conn.cursor() as cur:
                query = """
                    SELECT id, name, embedding, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                    FROM face_embeddings
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                """
                cur.execute(query, (encoding.tolist(), encoding.tolist(),0.6,encoding.tolist()))
                results = cur.fetchall()
                
                return [
                    {
                        "id": result[0],
                        "name": result[1],
                        "embedding": result[2],
                        "created_at": result[3],
                        "similarity": result[4]
                    }
                    for result in results
                ]
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []


    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()