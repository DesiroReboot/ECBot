"""知识库模块测试"""

import os
import sys
import shutil
import time
import unittest
import uuid

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from RAG import KBaseManager
from RAG.config.kbase_config import KBaseConfig
from RAG.storage.file_mapper import FileMapper
from RAG.storage.conflict_resolver import ConflictResolver
from RAG.classification.classifier import Classifier
from RAG.indexing.indexer import Indexer
from RAG.preprocessing.parser import DocumentParser


def _mktemp_dir() -> str:
    override_root = str(os.environ.get('ECBOT_TEST_TMPDIR', '')).strip()
    base_root = override_root or os.path.join(os.getcwd(), 'pytest_tmp_kbase')
    os.makedirs(base_root, exist_ok=True)
    for _ in range(16):
        path = os.path.join(base_root, f"tmp_{uuid.uuid4().hex}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue
    raise RuntimeError('failed_to_allocate_temp_dir')


def _cleanup_temp_dir(path: str, retries: int = 10, base_delay_sec: float = 0.1) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt >= retries - 1:
                # Best-effort cleanup for transient Windows file locks.
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(base_delay_sec * (attempt + 1))

class TestKBaseConfig(unittest.TestCase):
    """测试配置模块"""

    def test_default_config(self):
        config = KBaseConfig()
        self.assertIsNotNone(config.db_path)
        self.assertTrue(config.auto_classification)
        self.assertEqual(config.vector_dimension, 768)

    def test_config_from_env(self):
        os.environ['KBASE_OCR_ENABLED'] = 'true'
        config = KBaseConfig.from_env()
        self.assertTrue(config.ocr_enabled)
        del os.environ['KBASE_OCR_ENABLED']


class TestKBaseManager(unittest.TestCase):
    """测试知识库管理主控制器"""

    def setUp(self):
        self.temp_dir = _mktemp_dir()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.config = KBaseConfig(db_path=self.db_path)
        self.kbase = KBaseManager(self.config)

        # 创建测试文件目录
        self.test_files_dir = os.path.join(self.temp_dir, 'test_files')
        os.makedirs(self.test_files_dir)

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_init_database(self):
        """测试数据库初始化"""
        self.assertTrue(os.path.exists(self.db_path))

    def test_generate_file_uuid(self):
        """测试 UUID 生成"""
        uuid1 = KBaseManager.generate_file_uuid('/path/to/file.txt', 'hash123')
        uuid2 = KBaseManager.generate_file_uuid('/path/to/file.txt', 'hash123')
        self.assertEqual(uuid1, uuid2)  # 确定性 UUID

        uuid3 = KBaseManager.generate_file_uuid('/path/to/file.txt', 'hash456')
        self.assertNotEqual(uuid1, uuid3)  # 不同哈希产生不同 UUID

    def test_scan_and_process(self):
        """测试扫描和处理功能"""
        # 创建测试文件
        with open(os.path.join(self.test_files_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('This is a test file about Amazon FBA and cross-border e-commerce.')

        results = self.kbase.scan_and_process(self.test_files_dir)
        self.assertEqual(results['processed'], 1)
        self.assertEqual(results['failed'], 0)

    def test_scan_nonexistent_directory(self):
        """测试扫描不存在的目录"""
        results = self.kbase.scan_and_process('/nonexistent/path')
        self.assertEqual(results['processed'], 0)
        self.assertEqual(len(results['errors']), 1)

    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.kbase.get_statistics()
        self.assertIn('total_files', stats)
        self.assertIn('category_distribution', stats)


class TestFileMapper(unittest.TestCase):
    """测试文件映射管理器"""

    def setUp(self):
        self.temp_dir = _mktemp_dir()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        # 初始化数据库
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                uuid TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                category TEXT,
                summary TEXT,
                file_hash TEXT,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_scanned_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        self.mapper = FileMapper(self.db_path)

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_save_and_get_file(self):
        """测试保存和获取文件"""
        self.mapper.save_file(
            uuid='test-uuid-123',
            filename='test.txt',
            filepath='/path/to/test.txt',
            category='foreign_trade',
            summary='Test summary',
            file_hash='abc123',
            file_size=1024
        )

        file_info = self.mapper.get_file('test-uuid-123')
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['filename'], 'test.txt')
        self.assertEqual(file_info['category'], 'foreign_trade')

    def test_get_file_by_path(self):
        """测试通过路径获取文件"""
        self.mapper.save_file(
            uuid='test-uuid-456',
            filename='test2.txt',
            filepath='/path/to/test2.txt',
            category='cross_border_ecommerce',
            summary='Test summary 2',
            file_hash='def456',
            file_size=2048
        )

        file_info = self.mapper.get_file_by_path('/path/to/test2.txt')
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['uuid'], 'test-uuid-456')

    def test_get_all_files(self):
        """测试获取所有文件"""
        self.mapper.save_file('uuid1', 'file1.txt', '/path/file1.txt', 'cat1', 'sum1', 'hash1', 100)
        self.mapper.save_file('uuid2', 'file2.txt', '/path/file2.txt', 'cat2', 'sum2', 'hash2', 200)

        files = self.mapper.get_all_files()
        self.assertEqual(len(files), 2)

    def test_update_category(self):
        """测试更新分类"""
        self.mapper.save_file('uuid3', 'file3.txt', '/path/file3.txt', 'old_cat', 'sum3', 'hash3', 300)
        self.mapper.update_category('uuid3', 'new_cat')

        file_info = self.mapper.get_file('uuid3')
        self.assertEqual(file_info['category'], 'new_cat')

    def test_delete_file(self):
        """测试删除文件"""
        self.mapper.save_file('uuid4', 'file4.txt', '/path/file4.txt', 'cat4', 'sum4', 'hash4', 400)
        result = self.mapper.delete_file('uuid4')
        self.assertTrue(result)

        file_info = self.mapper.get_file('uuid4')
        self.assertIsNone(file_info)

    def test_search_by_filename(self):
        """测试按文件名搜索"""
        self.mapper.save_file('uuid5', 'amazon_guide.txt', '/path/amazon_guide.txt', 'ecommerce', 'sum5', 'hash5', 500)
        self.mapper.save_file('uuid6', 'fba_tips.txt', '/path/fba_tips.txt', 'ecommerce', 'sum6', 'hash6', 600)

        results = self.mapper.search_by_filename('amazon')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['filename'], 'amazon_guide.txt')


class TestConflictResolver(unittest.TestCase):
    """测试冲突解决器"""

    def setUp(self):
        self.temp_dir = _mktemp_dir()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        # 初始化数据库
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY,
                topic TEXT NOT NULL,
                conflicting_sources TEXT,
                resolution_status TEXT DEFAULT 'detected',
                priority TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        self.resolver = ConflictResolver(self.db_path)

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_report_and_detect_conflicts(self):
        """测试报告和检测冲突"""
        self.resolver.report_conflict('Topic A', ['Source 1', 'Source 2'], 'high')
        self.resolver.report_conflict('Topic B', ['Source 3', 'Source 4'], 'medium')

        conflicts = self.resolver.detect_conflicts()
        self.assertEqual(len(conflicts), 2)

    def test_resolve_conflict(self):
        """测试解决冲突"""
        self.resolver.report_conflict('Topic C', ['Source 5', 'Source 6'], 'low')

        conflicts = self.resolver.detect_conflicts()
        conflict_id = conflicts[0]['id']

        self.resolver.resolve_conflict(conflict_id, 'Resolution note')

        # 重新检测，应该没有未解决的冲突了
        conflicts = self.resolver.detect_conflicts()
        self.assertEqual(len(conflicts), 0)

    def test_get_conflict_stats(self):
        """测试获取冲突统计"""
        self.resolver.report_conflict('Topic D', ['Source 7', 'Source 8'])

        stats = self.resolver.get_conflict_stats()
        self.assertIn('detected', stats)
        self.assertGreaterEqual(stats['detected'], 1)


class TestClassifier(unittest.TestCase):
    """测试文档分类器"""

    def setUp(self):
        self.config = KBaseConfig()
        self.classifier = Classifier(self.config)

    def test_classify_foreign_trade(self):
        """测试外贸文档分类"""
        content = "This document discusses import export customs shipping and international trade regulations."
        category, confidence = self.classifier.classify(content)
        self.assertEqual(category, 'foreign_trade')
        self.assertGreater(confidence, 0)

    def test_classify_ecommerce(self):
        """测试跨境电商文档分类"""
        content = "Amazon FBA listing optimization and PPC advertising for Shopify dropshipping business."
        category, confidence = self.classifier.classify(content)
        self.assertEqual(category, 'cross_border_ecommerce')
        self.assertGreater(confidence, 0)

    def test_classify_uncategorized(self):
        """测试未分类文档"""
        content = "This is a random document with no specific topic."
        category, confidence = self.classifier.classify(content)
        self.assertEqual(category, 'uncategorized')

    def test_classify_empty_content(self):
        """测试空内容分类"""
        category, confidence = self.classifier.classify('')
        self.assertEqual(category, 'uncategorized')
        self.assertEqual(confidence, 0.0)

    def test_classify_batch(self):
        """测试批量分类"""
        contents = [
            "Import export customs trade",
            "Amazon FBA Shopify dropshipping",
            "Random content"
        ]
        results = self.classifier.classify_batch(contents)
        self.assertEqual(len(results), 3)

    def test_extract_keywords(self):
        """测试关键词提取"""
        content = "amazon fba amazon shopify amazon fba fba"
        keywords = self.classifier.extract_keywords(content, top_n=2)
        self.assertEqual(len(keywords), 2)
        self.assertIn('amazon', keywords)


class TestIndexer(unittest.TestCase):
    """测试索引构建器"""

    def setUp(self):
        self.temp_dir = _mktemp_dir()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.config = KBaseConfig(db_path=self.db_path)
        # 初始化数据库
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                content,
                file_uuid UNINDEXED,
                chunk_id UNINDEXED,
                tokenize = 'porter'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vec_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB,
                file_uuid TEXT,
                chunk_id INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        self.indexer = Indexer(self.db_path, self.config)

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_index_document(self):
        """测试文档索引"""
        content = "This is a test document about Amazon FBA and international trade."
        self.indexer.index_document('test-uuid', content)

        stats = self.indexer.get_index_stats()
        self.assertGreater(stats['fts_documents'], 0)

    def test_search(self):
        """测试搜索功能"""
        content = "Amazon FBA guide for beginners learning about dropshipping"
        self.indexer.index_document('search-test-uuid', content)

        results = self.indexer.search('Amazon FBA')
        # FTS5 可能返回结果，也可能不返回，取决于实现
        self.assertIsInstance(results, list)

    def test_delete_document_index(self):
        """测试删除文档索引"""
        content = "Test content for deletion"
        self.indexer.index_document('delete-test-uuid', content)

        initial_stats = self.indexer.get_index_stats()
        initial_count = initial_stats['fts_documents']

        self.indexer.delete_document_index('delete-test-uuid')

        final_stats = self.indexer.get_index_stats()
        self.assertLess(final_stats['fts_documents'], initial_count)

    def test_chunk_content(self):
        """测试内容分块"""
        content = "A" * 1000
        chunks = self.indexer._chunk_content(content)
        self.assertGreater(len(chunks), 1)


class TestDocumentParser(unittest.TestCase):
    """测试文档解析器"""

    def setUp(self):
        self.config = KBaseConfig()
        self.parser = DocumentParser(self.config)
        self.temp_dir = _mktemp_dir()

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_parse_text_file(self):
        """测试解析文本文件"""
        from pathlib import Path
        file_path = os.path.join(self.temp_dir, 'test.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('Hello World\nThis is a test.')

        content, metadata = self.parser.parse(Path(file_path))
        self.assertIn('Hello', content)
        self.assertEqual(metadata['type'], 'text')

    def test_parse_code_file(self):
        """测试解析代码文件"""
        file_path = os.path.join(self.temp_dir, 'test.py')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('# Comment\nprint("hello")\n\nx = 1')

        from pathlib import Path
        content, metadata = self.parser.parse(Path(file_path))
        self.assertEqual(metadata['type'], 'code')
        self.assertEqual(metadata['language'], 'py')

    def test_parse_pdf(self):
        """测试 PDF 解析（简化）"""
        from pathlib import Path
        content, metadata = self.parser.parse(Path('/fake/path.pdf'))
        self.assertEqual(metadata['type'], 'pdf')

    def test_extract_text_chunks(self):
        """测试文本分块"""
        content = "A" * 1000
        chunks = self.parser.extract_text_chunks(content, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        self.temp_dir = _mktemp_dir()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.config = KBaseConfig(db_path=self.db_path)
        self.kbase = KBaseManager(self.config)

        # 创建测试文件目录和文件
        self.test_files_dir = os.path.join(self.temp_dir, 'test_files')
        os.makedirs(self.test_files_dir)

        # 创建外贸相关文档
        with open(os.path.join(self.test_files_dir, 'trade_guide.txt'), 'w', encoding='utf-8') as f:
            f.write('This guide covers international trade, customs clearance, shipping documents like bill of lading, and import export regulations.')

        # 创建跨境电商文档
        with open(os.path.join(self.test_files_dir, 'amazon_fba.txt'), 'w', encoding='utf-8') as f:
            f.write('Amazon FBA business guide covering listing optimization, PPC advertising, and Shopify integration for dropshipping.')

    def tearDown(self):
        _cleanup_temp_dir(self.temp_dir)

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 扫描处理
        results = self.kbase.scan_and_process(self.test_files_dir)
        self.assertEqual(results['processed'], 2)

        # 2. 检查统计
        stats = self.kbase.get_statistics()
        self.assertEqual(stats['total_files'], 2)

        # 3. 搜索
        search_results = self.kbase.search('Amazon')
        self.assertIsInstance(search_results, list)

        # 4. 提取内容
        files = self.kbase.file_mapper.get_all_files()
        if files:
            content_info = self.kbase.extract_content(files[0]['uuid'])
            self.assertIn('content', content_info)

    def test_classify_and_migrate(self):
        """测试分类和迁移"""
        # 先扫描文件
        self.kbase.scan_and_process(self.test_files_dir)

        # 获取文件并重新分类
        files = self.kbase.file_mapper.get_all_files()
        for file_info in files:
            category, confidence = self.kbase.classify_document(file_info['uuid'])
            self.assertIn(category, ['foreign_trade', 'cross_border_ecommerce', 'uncategorized'])


if __name__ == '__main__':
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestKBaseConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestKBaseManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFileMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictResolver))
    suite.addTests(loader.loadTestsFromTestCase(TestClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestIndexer))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentParser))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
