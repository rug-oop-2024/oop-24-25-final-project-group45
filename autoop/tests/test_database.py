import random
import tempfile
import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        self.assertIsInstance(self.db, Database)

    def test_insert(self):
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.insert("collection", id, entry)
        self.assertEqual(self.db.fetch("collection", id)["key"], entry["key"])

    def test_remove(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.insert("collection", id, value)
        self.db.remove("collection", id)
        self.assertIsNone(self.db.fetch("collection", id))
        self.db.reload()
        self.assertIsNone(self.db.fetch("collection", id))

    def test_persistence(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.insert("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.fetch("collection", id)["key"], value["key"])

    def test_reload(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.insert("collection", key, value)
        other_db.reload()
        self.assertEqual(other_db.fetch("collection", key)["key"], value["key"])

    def test_list_entries(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.insert("collection", key, value)
        # Check that the key-value pair is in the collection
        self.assertIn((key, value), self.db.list_entries("collection"))
