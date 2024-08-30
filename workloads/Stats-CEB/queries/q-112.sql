SELECT COUNT(*)
FROM comments AS c,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  posts AS p
WHERE pl.PostId = p.Id
  AND c.PostId = p.Id
  AND v.PostId = p.Id
  AND ph.PostId = p.Id
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-06-14 13:31:35' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-10 00:00:00' AS timestamp);
