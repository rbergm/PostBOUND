SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  posts AS p
WHERE ph.PostId = p.Id
  AND c.PostId = p.Id
  AND v.PostId = p.Id
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp);
